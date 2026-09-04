//! Four crawlers, one host, one rate.
//!
//! This is the test the whole lease design exists for. Without an authority,
//! N nodes each holding pages of one site each wait their own delay and the
//! site receives N times what it asked for. With one, the requests interleave
//! into a single spaced sequence however many crawlers there are.
//!
//! What is asserted here and what is not:
//!
//! Everything crossing the wire carries a *relative* wait, which each client
//! turns into an absolute moment using its own clock after the reply arrives.
//! Two such moments differ by the round-trip jitter between them, measured at
//! 11 ms on a loaded machine against a 30 ms floor — so no assertion here
//! compares two of them, and none measures elapsed wall-clock to derive a
//! spacing. What is asserted is the exact values a lease carries, and the
//! aggregate span, which carries that jitter once instead of once per gap.
//!
//! The exact spacing contract belongs to `LocalPolicy`, which returns absolute
//! instants from one clock with no conversion, and is tested there in
//! `indexander_crawl::politeness`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use indexander_cluster::leases::{LeaseAuthority, remote_politeness};
use indexander_crawl::politeness::{Politeness, wait_for};
use tokio::net::TcpListener;

/// Starts an authority with the given floor, returns its address.
async fn authority(floor: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let authority = Arc::new(LeaseAuthority::new(floor));
    tokio::spawn(async move {
        let _ = authority.serve(listener).await;
    });
    address
}

#[tokio::test]
async fn one_crawler_is_spaced_by_the_floor() {
    let address = authority(Duration::from_millis(40)).await;
    let politeness = remote_politeness(&address).await.expect("connect");

    let started = Instant::now();
    for _ in 0..4 {
        wait_for(politeness.lease("example.com", Duration::ZERO, 1).await).await;
    }
    // Four requests at 40 ms apart: the first is immediate, so three gaps.
    assert!(
        started.elapsed() >= Duration::from_millis(120),
        "four requests took only {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn four_crawlers_sharing_an_authority_do_not_multiply_the_rate() {
    let floor = Duration::from_millis(30);
    let address = authority(floor).await;

    // Four independent crawlers, as four separate connections would be four
    // separate machines.
    let mut handles = Vec::new();
    let started = Instant::now();
    for _ in 0..4 {
        let address = address.clone();
        handles.push(tokio::spawn(async move {
            let politeness = remote_politeness(&address).await.expect("connect");
            let mut granted = Vec::new();
            for _ in 0..3 {
                let lease = politeness.lease("shared.example", Duration::ZERO, 1).await;
                // The moment the authority granted, not the moment this task
                // happened to wake up. The authority controls the first; the
                // operating system's scheduler controls the second, and a
                // loaded machine can compress two wake-ups into one instant
                // without anything being wrong.
                granted.push(lease.not_before);
                wait_for(lease).await;
            }
            granted
        }));
    }

    let mut granted: Vec<Instant> = Vec::new();
    for handle in handles {
        granted.extend(handle.await.expect("crawler"));
    }
    granted.sort_unstable();
    assert_eq!(granted.len(), 12);

    // Asserted in aggregate, not gap by gap.
    //
    // Each client reconstructs an absolute instant from a *relative* wait,
    // using its own clock, after the reply arrives — so the difference in
    // round-trip time between two requests lands directly in the difference
    // between two reconstructed instants. On a loaded machine that was
    // measured at 11 ms against a 30 ms floor, which makes any per-gap
    // assertion here a test of the runner rather than of the code.
    //
    // The span across all twelve carries the same jitter once, not twelve
    // times, and it is what actually separates the two outcomes: coordinated,
    // the twelve slots span 330 ms; uncoordinated, four crawlers each pacing
    // only themselves fit three requests each into 60 ms. Anything above 250
    // ms can only be the coordinated case.
    //
    // The exact per-slot contract is asserted against the authority itself in
    // `the_authority_never_grants_overlapping_slots`, where there is no wire
    // and no clock to transfer.
    let span = granted
        .last()
        .expect("last")
        .duration_since(*granted.first().expect("first"));
    assert!(
        span >= Duration::from_millis(250),
        "twelve slots spanned {span:?}; uncoordinated crawlers would fit them in about 60 ms"
    );
    let _ = started;
}

#[tokio::test]
async fn different_hosts_are_paced_independently() {
    let address = authority(Duration::from_millis(100)).await;
    let politeness = remote_politeness(&address).await.expect("connect");

    // Asserted on what the leases say rather than on the wall clock: a
    // machine under load can make any timing test fail without anything
    // being wrong, and "was granted immediately" is the actual property.
    for host in ["a.example", "b.example", "c.example", "d.example"] {
        let lease = politeness.lease(host, Duration::ZERO, 1).await;
        let wait = lease.not_before.saturating_duration_since(Instant::now());
        assert!(
            wait < Duration::from_millis(10),
            "{host} was made to wait {wait:?} behind another host"
        );
    }
}

#[tokio::test]
async fn the_authority_enforces_its_floor_over_a_smaller_request() {
    // A crawler asking for no delay does not get no delay.
    let address = authority(Duration::from_millis(50)).await;
    let politeness = remote_politeness(&address).await.expect("connect");

    // The granted spacing is the contract, and it is an exact value rather
    // than an observation: asserting on elapsed wall-clock here would measure
    // the round trip, not the floor.
    let first = politeness.lease("greedy.example", Duration::ZERO, 1).await;
    assert_eq!(first.spacing, Duration::from_millis(50));

    // And the second lease is pushed out behind the first.
    let second = politeness.lease("greedy.example", Duration::ZERO, 1).await;
    assert!(
        second.not_before > first.not_before,
        "a second request to the same host was granted the same slot"
    );
}

#[tokio::test]
async fn a_host_asking_for_more_gets_more() {
    let address = authority(Duration::from_millis(10)).await;
    let politeness = remote_politeness(&address).await.expect("connect");
    let lease = politeness
        .lease("slow.example", Duration::from_millis(200), 1)
        .await;
    assert_eq!(lease.spacing, Duration::from_millis(200));
}

#[tokio::test]
async fn a_crawl_survives_the_authority_disappearing() {
    // A rate limiter falling over must not stop a crawl. It falls back to the
    // crawler's own delay, which is what a single-node crawl would have used.
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let authority = Arc::new(LeaseAuthority::new(Duration::from_millis(10)));
    let task = tokio::spawn(async move {
        let _ = authority.serve(listener).await;
    });

    let politeness = remote_politeness(&address).await.expect("connect");
    let _ = politeness.lease("example.com", Duration::ZERO, 1).await;

    task.abort();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let lease = politeness
        .lease("example.com", Duration::from_millis(70), 1)
        .await;
    assert_eq!(lease.spacing, Duration::from_millis(70));
}

#[tokio::test]
async fn a_local_policy_and_a_remote_one_behave_the_same() {
    // The property that makes the abstraction worth having.
    let floor = Duration::from_millis(25);
    let address = authority(floor).await;

    for politeness in [
        Politeness::local(floor),
        remote_politeness(&address).await.expect("connect"),
    ] {
        let started = Instant::now();
        for _ in 0..3 {
            wait_for(politeness.lease("same.example", Duration::ZERO, 1).await).await;
        }
        // Only the lower bound is asserted. Being slower than the floor is
        // never a bug — a loaded machine, a busy authority — and asserting an
        // upper bound on wall-clock time makes a test that fails for reasons
        // that have nothing to do with the code.
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "three requests took {:?}, faster than the floor allows",
            started.elapsed()
        );
    }
}
