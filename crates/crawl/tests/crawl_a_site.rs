//! End-to-end crawls against a throwaway HTTP server on localhost.
//!
//! No test here touches the network. The server is a few dozen lines of
//! `tokio` that answers from a fixed table, which makes the whole crawl
//! deterministic: the same seed produces the same pages, every run.

use std::collections::HashMap;
use std::time::Duration;

use indexander_crawl::frontier::Limits;
use indexander_crawl::{Config, crawl_to_vec};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;

/// Serves `routes` until dropped. Returns the base URL it is listening on.
async fn serve(
    routes: HashMap<&'static str, (&'static str, &'static str)>,
) -> (Url, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let base = Url::parse(&format!("http://{addr}/")).expect("base url");

    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let routes = routes.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let Ok(n) = socket.read(&mut buf).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = request.split_whitespace().nth(1).unwrap_or("/").to_owned();

                let response = match routes.get(path.as_str()) {
                    Some((content_type, body)) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    ),
                    None => {
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_owned()
                    }
                };
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    (base, handle)
}

fn fast_config() -> Config {
    Config {
        // No politeness delay: the "host" is this process.
        delay: Duration::from_millis(0),
        timeout: Duration::from_secs(5),
        concurrency: 2,
        limits: Limits {
            max_pages: 50,
            max_depth: 3,
            max_pages_per_host: 50,
            same_host_only: true,
        },
        ..Config::default()
    }
}

fn site() -> HashMap<&'static str, (&'static str, &'static str)> {
    HashMap::from([
        (
            "/robots.txt",
            ("text/plain", "User-agent: *\nDisallow: /private\n"),
        ),
        (
            "/",
            (
                "text/html",
                r#"<html><head><title>Inicio</title></head><body>
                   <p>Portada del sitio.</p>
                   <a href="/parasearch">el buscador colombiano</a>
                   <a href="/private/secreto">no entrar</a>
                   <a href="/acerca.html">acerca de</a>
                   <a href="https://otro-dominio.example/x">fuera</a>
                   </body></html>"#,
            ),
        ),
        (
            "/parasearch",
            (
                "text/html",
                r#"<html><head><title>Parasearch</title></head><body>
                   <p>Un motor de b&uacute;squeda escrito en Perl en 2004.</p>
                   <a href="/acerca.html">acerca de</a>
                   </body></html>"#,
            ),
        ),
        (
            "/acerca.html",
            (
                "text/html",
                "<html><head><title>Acerca</title></head><body><p>Compa&ntilde;ia.</p></body></html>",
            ),
        ),
        (
            "/private/secreto",
            ("text/html", "<html><title>Secreto</title></html>"),
        ),
    ])
}

#[tokio::test]
async fn crawls_a_site_and_obeys_its_rules() {
    let (base, server) = serve(site()).await;
    let (docs, stats) = crawl_to_vec(&fast_config(), std::slice::from_ref(&base))
        .await
        .expect("crawl should run");
    server.abort();

    let uris: Vec<String> = docs.iter().map(|d| d.uri.clone()).collect();

    // Three reachable, allowed pages.
    assert_eq!(docs.len(), 3, "got {uris:?}");
    assert!(uris.iter().any(|u| u.ends_with('/')));
    assert!(uris.iter().any(|u| u.ends_with("/parasearch")));
    assert!(uris.iter().any(|u| u.ends_with("/acerca.html")));

    // robots.txt said no, so it was never fetched.
    assert!(
        !uris.iter().any(|u| u.contains("private")),
        "crawled a disallowed path: {uris:?}"
    );
    assert_eq!(stats.disallowed_by_robots, 1);

    // A link off-host is not followed when same_host_only is set.
    assert!(!uris.iter().any(|u| u.contains("otro-dominio")));

    assert_eq!(stats.indexed, 3);
    assert_eq!(stats.errors, 0);
}

#[tokio::test]
async fn anchor_text_from_the_linking_page_lands_on_the_linked_page() {
    let (base, server) = serve(site()).await;
    let (docs, _) = crawl_to_vec(&fast_config(), &[base]).await.expect("crawl");
    server.abort();

    let parasearch = docs
        .iter()
        .find(|d| d.uri.ends_with("/parasearch"))
        .expect("parasearch page");

    // The words came from the home page, not from this one.
    assert!(
        parasearch
            .anchors
            .iter()
            .any(|a| a == "el buscador colombiano"),
        "anchors were {:?}",
        parasearch.anchors
    );
    assert!(!parasearch.body.contains("el buscador colombiano"));

    // Two pages link to /acerca.html with the same words; they collapse.
    let acerca = docs
        .iter()
        .find(|d| d.uri.ends_with("/acerca.html"))
        .expect("acerca page");
    assert_eq!(acerca.anchors, ["acerca de"]);
}

#[tokio::test]
async fn entities_survive_the_whole_pipeline() {
    let (base, server) = serve(site()).await;
    let (docs, _) = crawl_to_vec(&fast_config(), &[base]).await.expect("crawl");
    server.abort();

    let acerca = docs
        .iter()
        .find(|d| d.uri.ends_with("/acerca.html"))
        .unwrap();
    assert!(
        acerca.body.contains("Compañia"),
        "body was {:?}",
        acerca.body
    );

    let parasearch = docs
        .iter()
        .find(|d| d.uri.ends_with("/parasearch"))
        .unwrap();
    assert!(parasearch.body.contains("búsqueda"));
    assert_eq!(parasearch.title, "Parasearch");
}

#[tokio::test]
async fn depth_zero_fetches_only_the_seed() {
    let (base, server) = serve(site()).await;
    let config = Config {
        limits: Limits {
            max_depth: 0,
            ..fast_config().limits
        },
        ..fast_config()
    };
    let (docs, _) = crawl_to_vec(&config, &[base]).await.expect("crawl");
    server.abort();
    assert_eq!(docs.len(), 1);
}

#[tokio::test]
async fn the_page_budget_is_respected() {
    let (base, server) = serve(site()).await;
    let config = Config {
        limits: Limits {
            max_pages: 2,
            ..fast_config().limits
        },
        ..fast_config()
    };
    let (docs, stats) = crawl_to_vec(&config, &[base]).await.expect("crawl");
    server.abort();
    assert!(docs.len() <= 2, "budget of 2 produced {}", docs.len());
    assert!(stats.fetched <= 2);
}

#[tokio::test]
async fn a_host_that_cannot_answer_is_not_crawled() {
    // Nothing is listening on this port, so robots.txt fails with a network
    // error. Being unable to ask is not permission.
    let dead = Url::parse("http://127.0.0.1:1/").expect("url");
    let (docs, stats) = crawl_to_vec(&fast_config(), &[dead]).await.expect("crawl");
    assert!(docs.is_empty());
    assert_eq!(stats.fetched, 0);
    assert_eq!(stats.disallowed_by_robots, 1);
}

#[tokio::test]
async fn a_site_with_no_robots_txt_is_crawled_normally() {
    let mut routes = site();
    routes.remove("/robots.txt"); // now a 404
    let (base, server) = serve(routes).await;
    let (docs, stats) = crawl_to_vec(&fast_config(), &[base]).await.expect("crawl");
    server.abort();
    // Including /private/secreto, since nothing forbids it any more.
    assert_eq!(
        docs.len(),
        4,
        "got {:?}",
        docs.iter().map(|d| &d.uri).collect::<Vec<_>>()
    );
    assert_eq!(stats.disallowed_by_robots, 0);
}

#[tokio::test]
async fn non_text_responses_are_skipped_not_indexed() {
    let routes = HashMap::from([
        ("/robots.txt", ("text/plain", "")),
        (
            "/",
            ("text/html", r#"<a href="/imagen.png">una imagen</a>"#),
        ),
        ("/imagen.png", ("image/png", "PNG-ish bytes")),
    ]);
    let (base, server) = serve(routes).await;
    let (docs, stats) = crawl_to_vec(&fast_config(), &[base]).await.expect("crawl");
    server.abort();

    assert_eq!(docs.len(), 1, "only the html page should be indexed");
    assert_eq!(stats.skipped_content_type, 1);
}
