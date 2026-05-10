# References

## Real-Time Scheduling

1. Liu, C. L., & Layland, J. W. (1973). "Scheduling algorithms for multiprogramming in a hard-real-time environment." *Journal of the ACM*, 20(1), 46–61. [Theorem 5.2]

2. Buttazzo, G. C. (2011). *Hard Real-Time Computing Systems* (3rd ed.). Springer. [Section 5.5.1 — Synchronous Busy Period]

3. Yermakou, D. (2026). "AxonOS: Analytical Real-Time Schedulability of a Safety-Critical BCI Microkernel." *arXiv preprint*. [Proposition 5.4]

## Memory Ordering

4. Vyukov, D. (2010). "Lock-free algorithms: The queue and the ring buffer." *Dmitry Vyukov's Blog*. [Sequence-number protocol]

5. Batty, M., Owens, S., Sarkar, S., Sewell, P., & Weber, T. (2011). "Mathematizing C++ concurrency." *POPL 2011*. [Release-Acquire semantics]

6. AxonOS RFC-0007 (2026). "SPSC Sequence-Number Correctness Proof (Theorem 6.3)." *axonos-rfcs*.

## Capability Security

7. Miller, M. S., Yee, K., & Shapiro, J. (2003). "Capability myths demolished." *SRL Technical Report*.

8. AxonOS RFC-0004 (2026). "Structural Data Minimisation via Type System (Theorem 8.3)." *axonos-rfcs*.

## BCI Signal Processing

9. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K. R. (2008). "Optimizing spatial filters for robust EEG single-trial analysis." *IEEE Signal Processing Magazine*, 25(1), 41–56. [CSP]

10. Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). "Optimal spatial filtering of single trial EEG during imagined hand movement." *IEEE TBME*, 47(4), 583–584. [CSP for motor imagery]

11. Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition* (2nd ed.). Academic Press. [LDA]

12. Welch, G., & Bishop, G. (2006). "An Introduction to the Kalman Filter." *UNC-Chapel Hill Technical Report TR 95-041*. [Kalman]

## Information Theory / Privacy

13. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. [Theorem 9.1]

14. AxonOS RFC-0005 (2026). "Min-Entropy Residual Bounds (Theorem 9.3)." *axonos-rfcs*.

## Riemannian Geometry

15. Congedo, M., Barachant, A., & Bhatia, R. (2017). "Riemannian geometry for EEG-based brain-computer interfaces." *IEEE TBCI*.

16. Yger, F. (2013). "A review of classification algorithms for EEG-based brain-computer interfaces." *Journal of Neural Engineering*, 10(3).

## Regulatory

17. IEC 62304:2006+AMD1:2015. "Medical device software — Software life cycle processes." [Class C alignment]

18. FDA (2023). "Content of Premarket Submissions for Management of Cybersecurity in Medical Devices." *Guidance for Industry*.

## Hardware

19. STMicroelectronics (2024). *RM0090 Reference Manual — STM32F4xx*. [DWT, NVIC, GPIO]

20. Texas Instruments (2023). *ADS1299 datasheet — 8-Channel 24-Bit ADC*. [250 SPS, SPI interface]

21. Microchip (2024). *ATECC608B datasheet — CryptoAuthentication Device*. [HMAC-SHA256]
