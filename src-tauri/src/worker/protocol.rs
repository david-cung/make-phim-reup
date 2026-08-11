//! JSON-RPC 2.0 line-delimited protocol shared by Rust and the Python worker.
//!
//! Two kinds of frames come back from the worker on stdout:
//!
//! * **Responses** (`{"id": ..., "result"|"error": ...}`) — one per
//!   completed request.
//! * **Notifications** (`{"method": ..., "params": ...}`, no `id`) —
//!   free-form messages the worker can emit at any time. Phase 3 uses
//!   them for `stt.progress` and `stt.download_progress`.
//!
//! [`RpcFrame::parse_line`] classifies an incoming line without losing
//! either shape.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub type RpcRequestId = String;

/// A single request written to the worker's stdin, one line per request.
#[derive(Debug, Clone, Serialize)]
pub struct RpcRequest {
    pub jsonrpc: &'static str,
    pub id: RpcRequestId,
    pub method: String,
    pub params: Value,
}

impl RpcRequest {
    pub fn new(id: impl Into<RpcRequestId>, method: impl Into<String>, params: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}

/// A notification frame sent from Rust → worker (no ``id``).
#[derive(Debug, Clone, Serialize)]
pub struct RpcNotification<'a> {
    pub jsonrpc: &'static str,
    pub method: &'a str,
    pub params: Value,
}

impl<'a> RpcNotification<'a> {
    pub fn new(method: &'a str, params: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            method,
            params,
        }
    }
}

/// One response line read from the worker's stdout.
#[derive(Debug, Clone, Deserialize)]
pub struct RpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: RpcRequestId,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<RpcError>,
}

/// A notification frame read from the worker's stdout.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingNotification {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// Classified incoming frame from the worker.
#[derive(Debug, Clone)]
pub enum RpcFrame {
    Response(RpcResponse),
    Notification(IncomingNotification),
}

impl RpcFrame {
    pub fn parse_line(line: &str) -> Result<Self, serde_json::Error> {
        let value: Value = serde_json::from_str(line)?;
        let has_id = value.get("id").is_some_and(|v| !v.is_null());
        if has_id {
            let resp: RpcResponse = serde_json::from_value(value)?;
            Ok(RpcFrame::Response(resp))
        } else {
            let notif: IncomingNotification = serde_json::from_value(value)?;
            Ok(RpcFrame::Notification(notif))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
#[error("{code}: {message}")]
pub struct RpcError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialises_as_line() {
        let r = RpcRequest::new("1", "ping", serde_json::json!({}));
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""jsonrpc":"2.0""#));
        assert!(s.contains(r#""method":"ping""#));
    }

    #[test]
    fn frame_parses_response() {
        let s = r#"{"jsonrpc":"2.0","id":"1","result":{"pong":true}}"#;
        match RpcFrame::parse_line(s).unwrap() {
            RpcFrame::Response(r) => {
                assert_eq!(r.id, "1");
                assert!(r.result.is_some());
                assert!(r.error.is_none());
            }
            RpcFrame::Notification(_) => panic!("expected response"),
        }
    }

    #[test]
    fn frame_parses_error_response() {
        let s = r#"{"jsonrpc":"2.0","id":"1","error":{"code":"E_METHOD","message":"unknown"}}"#;
        match RpcFrame::parse_line(s).unwrap() {
            RpcFrame::Response(r) => {
                assert!(r.result.is_none());
                assert_eq!(r.error.unwrap().code, "E_METHOD");
            }
            _ => panic!("expected response"),
        }
    }

    #[test]
    fn frame_parses_notification() {
        let s = r#"{"jsonrpc":"2.0","method":"stt.progress","params":{"requestId":"x","fraction":0.5}}"#;
        match RpcFrame::parse_line(s).unwrap() {
            RpcFrame::Notification(n) => {
                assert_eq!(n.method, "stt.progress");
                assert_eq!(n.params["requestId"], "x");
            }
            _ => panic!("expected notification"),
        }
    }

    #[test]
    fn frame_treats_null_id_as_notification() {
        let s = r#"{"jsonrpc":"2.0","id":null,"method":"noise","params":{}}"#;
        match RpcFrame::parse_line(s).unwrap() {
            RpcFrame::Notification(n) => assert_eq!(n.method, "noise"),
            _ => panic!("expected notification"),
        }
    }
}
