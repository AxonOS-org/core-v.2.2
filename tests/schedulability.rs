//! Schedulability tests (Liu-Layland)

use axonos_kernel::scheduler::{Task, TaskId, AdmissionController};

#[test]
fn test_liu_layland_bound() {
    let tasks = [
        Task::new(TaskId(0), 4000, 400),  // U = 0.10
        Task::new(TaskId(1), 4000, 400),  // U = 0.10
        Task::new(TaskId(2), 4000, 200),  // U = 0.05
    ];
    assert!(AdmissionController::is_schedulable(&tasks));
    assert_eq!(AdmissionController::busy_period_bound(&tasks), Some(1000));
}

#[test]
fn test_busy_period_convergence() {
    let tasks = [
        Task::new(TaskId(0), 1000, 200),
        Task::new(TaskId(1), 2000, 300),
    ];
    let l = AdmissionController::busy_period_bound(&tasks);
    assert!(l.is_some());
    assert!(l.unwrap() >= 500);
}
