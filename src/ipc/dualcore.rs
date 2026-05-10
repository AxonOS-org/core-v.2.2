//! Dual-Core Contract Implementation
//!
//! Partition model:
//! - M4F (hard real-time): signal pipeline, consent state machine, stimulation interlock
//! - A53 (soft real-time): session management, BLE/Wi-Fi egress, WebAssembly sandbox
//! - Shared SRAM: 64-slot SPSC ring buffer (64 bytes/slot, 4096 bytes total)
//!
//! # Safety
//! This struct must reside in shared SRAM with cache disabled or explicitly
//! flushed. Use `#[repr(C)]` and linker section placement.
//!
//! Reference: AxonOS RFC-0006, Section 4.2.

use super::{DcClause, ContractViolation};
use crate::ringbuf::SpscRingBuffer;
use crate::config;
use core::sync::atomic::{AtomicU64, Ordering};

/// Dual-core contract state machine
#[repr(C)]
pub struct DualCoreContract {
    clauses: [ClauseStatus; 6],
    shared_buffer: SpscRingBuffer<IntentPacket>,
    heartbeat_count: AtomicU64,
    last_heartbeat: AtomicU64,
    safe_idle_active: bool,
    violations: heapless::Vec<ContractViolation, 16>,
}

/// Intent packet for IPC
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IntentPacket {
    pub class: u8,
    pub confidence: u8,
    pub hmac_tag: [u8; 4],
    pub epoch: u64,
    pub timestamp: u64,
}

/// Clause status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseStatus {
    Satisfied,
    Monitoring,
    Violated,
    Pending,
}

impl DualCoreContract {
    pub fn new() -> Self {
        Self {
            clauses: [ClauseStatus::Pending; 6],
            shared_buffer: SpscRingBuffer::new(),
            heartbeat_count: AtomicU64::new(0),
            last_heartbeat: AtomicU64::new(0),
            safe_idle_active: false,
            violations: heapless::Vec::new(),
        }
    }

    pub fn send_intent(&self, packet: IntentPacket) -> Result<(), crate::ringbuf::SpscError> {
        self.shared_buffer.try_push(packet)
    }

    pub fn receive_intent(&self) -> Result<IntentPacket, crate::ringbuf::SpscError> {
        self.shared_buffer.try_pop()
    }

    pub fn send_heartbeat(&self, timestamp: u64) {
        self.heartbeat_count.fetch_add(1, Ordering::Relaxed);
        self.last_heartbeat.store(timestamp, Ordering::Release);
    }

    pub fn check_heartbeat(&mut self, now: u64) -> bool {
        let last = self.last_heartbeat.load(Ordering::Acquire);
        let elapsed_ms = now.saturating_sub(last) / 1000;
        if elapsed_ms > config::SAFE_IDLE_TIMEOUT_MS as u64 {
            self.safe_idle_active = true;
            self.clauses[4] = ClauseStatus::Violated;
            let _ = self.violations.push(ContractViolation {
                clause: DcClause::Dc5,
                timestamp: now,
                observed: Some(elapsed_ms as f32),
                expected: Some(config::SAFE_IDLE_TIMEOUT_MS as f32),
            });
            false
        } else {
            true
        }
    }

    pub fn is_safe_idle(&self) -> bool {
        self.safe_idle_active
    }

    pub fn clause_status(&self, clause: DcClause) -> ClauseStatus {
        match clause {
            DcClause::Dc1 => self.clauses[0],
            DcClause::Dc2 => self.clauses[1],
            DcClause::Dc3 => self.clauses[2],
            DcClause::Dc4 => self.clauses[3],
            DcClause::Dc5 => self.clauses[4],
            DcClause::Dc6 => self.clauses[5],
        }
    }

    pub fn violations(&self) -> &[ContractViolation] {
        &self.violations
    }

    pub fn reset(&mut self) {
        self.heartbeat_count.store(0, Ordering::Relaxed);
        self.last_heartbeat.store(0, Ordering::Relaxed);
        self.safe_idle_active = false;
        self.violations.clear();
    }
}
