# Capability Model

## Theorem 8.3

Structural data minimisation: no prohibited types reach applications.

## Manifest

Applications declare capabilities at install time. Kernel verifies M ⊆ K.

## Rate Limiting

| Capability | Max Rate |
|------------|----------|
| Navigation | 10 Hz |
| TextEntry | 5 Hz |
| Environmental | 2 Hz |
| Stimulation | 1 Hz |
| RawEeg | 250 Hz (restricted) |

## References

- Miller, M. S., Yee, K., & Shapiro, J. (2003). "Capability myths demolished." SRL TR.
- AxonOS RFC-0004: Capability Model Specification.
