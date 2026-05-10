//! Kernel configuration constants derived from hardware specification

/// CPU frequency [Hz] — STM32F407 at 168 MHz [L1]
pub const CPU_HZ: u32 = 168_000_000;

/// ADC sampling rate [SPS] — ADS1299 at 250 SPS [L1]
pub const ADC_SPS: u32 = 250;

/// Epoch period [µs] = 1 / 250 SPS = 4000 µs [L1]
pub const EPOCH_US: u32 = 4_000;

/// DWT cycle counter resolution [ns] ≈ 5.95 ns [L1]
pub const DWT_RESOLUTION_NS: f32 = 5.95;

/// Conservative admission ceiling U_max = 0.25 [L1]
pub const ADMISSION_CEILING: f32 = 0.25;

/// Number of EEG channels [L1]
pub const EEG_CHANNELS: usize = 8;

/// ADC resolution [bits] [L1]
pub const ADC_RESOLUTION: usize = 24;

/// Bytes per sample frame: 8 ch × 3 bytes = 24 bytes [L1]
pub const SAMPLE_FRAME_BYTES: usize = 24;

/// SPI DMA transactions per frame: ceil(24/4) = 6 [L1]
pub const SPI_DMA_TRANSACTIONS: usize = 6;

/// FIR filter order [L1]
pub const FIR_ORDER: usize = 64;

/// SPSC ring buffer capacity — power of 2 [L1]
pub const RING_BUFFER_CAPACITY: usize = 64;

/// Shared SRAM size for IPC [bytes] [L1]
pub const SHARED_SRAM_BYTES: usize = 4096;

/// HMAC-SHA256 tag length [bytes] [L1]
pub const HMAC_TAG_LEN: usize = 32;

/// Maximum tasks in EDF scheduler [L1]
pub const MAX_TASKS: usize = 8;

/// DC5 safe-idle timeout [ms] [L2]
pub const SAFE_IDLE_TIMEOUT_MS: u32 = 12;

/// A53 wake-up deterministic bound [µs] [L2]
pub const A53_WAKE_US: u32 = 50;

/// Flash wait states at 168 MHz [L1]
pub const FLASH_WAIT_STATES: u8 = 5;

/// PLL configuration: HSE 8 MHz → 168 MHz [L1]
/// PLLM = 8, PLLN = 336, PLLP = 2, PLLQ = 7
pub const PLL_CONFIG: u32 = 0x0740_3408;
