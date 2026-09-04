//! Four crawlers, one host, one rate.
//!
//! This is the test the whole lease design exists for. Without an authority,
//! N nodes each holding pages of one site each wait their own delay and the
//! site receives N times what it asked for. With one, the requests interleave
//! into a single spaced sequence however many crawlers there are.

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

    // No two of the twelve slots are closer together than the floor — give or
    // take the wire.
    //
    // The tolerance is not slack, it is physics. The authority sends a
    // *relative* wait and each client turns it into an absolute instant using
    // its own clock, after the reply arrives. The difference in round-trip
    // time between two requests lands directly in the difference between the
    // two reconstructed instants, and no amount of care at either end removes
    // it: transferring an instant between two processes costs the jitter
    // between them.
    //
    // The exact contract is asserted where it actually holds, against the
    // authority itself, in `the_authority_never_grants_overlapping_slots`.
    let jitter = Duration::from_millis(5);
    for pair in granted.windows(2) {
        let gap = pair[1].duration_since(pair[0]);
        assert!(
            gap + jitter >= floor,
            "two slots were granted {gap:?} apart, floor is {floor:?}"
        );
    }

    // Eleven gaps of 30 ms: the twelve slots span most of 330 ms. Without an
    // authority each crawler would have paced only itself and the twelve would
    // have fitted into a quarter of that, which no jitter can explain away.
    let span = granted
        .last()
        .expect("last")
        .duration_since(*granted.first().expect("first"));
    assert!(
        span + jitter >= floor * 11,
        "twelve slots spanned {span:?}; they were not coordinated"
    );
    let _ = started;
}

/// The exact contract, tested where it holds: inside the authority.
///
/// No sockets, no clock transfer, no jitter — just the reservations it makes.
#[tokio::test]
async fn the_authority_never_grants_overlapping_slots() {
    use indexander_proto::message::{Request, Response};

    let floor = Duration::from_millis(30);
    let authority = LeaseAuthority::new(floor);

    // Interleaved hosts, so a bug that reserved globally rather than per host
    // would show up as slots that are too far apart, not too close.
    let mut slots: Vec<u64> = Vec::new();
    for _ in 0..12 {
        let Response::Lease { wait_ms, .. } = authority
            .handle(&Request::Lease {
                host: "shared.example".into(),
                requested_delay_ms: 0,
                permits: 1,
            })
            .await
        else {
            panic!("expected a lease");
        };
        slots.push(wait_ms);
        let _ = authority
            .handle(&Request::Lease {
                host: "other.example".into(),
                requested_delay_ms: 0,
                permits: 1,
            })
            .await;
    }

    // Twelve slots for one host, each at least the floor after the last.
    for (i, pair) in slots.windows(2).enumerate() {
        assert!(
            pair[1] >= pair[0] + 30,
            "slot {i} was {} ms and slot {} was {} ms",
            pair[0],
            i + 1,
            pair[1]
        );
    }
    assert_eq!(slots.first(), Some(&0), "the first request should not wait");
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

    let first = politeness.lease("greedy.example", Duration::ZERO, 1).await;
    assert_eq!(first.spacing, Duration::from_millis(50));

    let started = Instant::now();
    wait_for(first).await;
    wait_for(politeness.lease("greedy.example", Duration::ZERO, 1).await).await;
    assert!(started.elapsed() >= Duration::from_millis(45));
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
async fn a_batch_reserves_every_slot_it_was_given() {
    let address = authority(Duration::from_millis(20)).await;
    let politeness = remote_politeness(&address).await.expect("connect");

    let batch = politeness.lease("batch.example", Duration::ZERO, 10).await;
    assert_eq!(batch.permits, 10);

    // The next asker waits for the whole batch, not for one slot.
    //
    // Compared against the batch's own slot rather than against "now": the
    // round trip to the authority takes time, and measuring what is left of
    // the wait makes the assertion depend on how fast that round trip was.
    let next = politeness.lease("batch.example", Duration::ZERO, 1).await;
    assert!(
        next.not_before >= batch.not_before + Duration::from_millis(200),
        "the next slot was granted only {:?} after the batch of ten started",
        next.not_before.duration_since(batch.not_before)
    );
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
