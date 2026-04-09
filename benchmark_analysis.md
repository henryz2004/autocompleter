# LLM Provider Benchmark Analysis

## Executive Summary

This analysis evaluates 24 provider/model combinations across 1008 benchmark runs (v2 + v3 + v4 combined). Each configuration was tested across multiple scenarios to measure:
- **Quality**: Correctness of generated suggestions (0-1 scale, 1.0 = perfect)
- **Latency**: Response time in milliseconds (avg, median, p95)
- **Reliability**: Failure rates and consistency

The benchmark simulates real-world autocomplete scenarios including continuation, reply, and code completion tasks.

## Results by Performance Tier

### Tier 1: Production-Ready
**Criteria**: Quality >= 0.98, Avg Latency < 500ms

| Provider | Model | Quality | Avg Latency | Median | p95 | Notes |
|---|---|---|---|---|---|---|
| Cerebras | Qwen3-235B | 1.000 | 254ms | 256ms | 289ms | **Best overall** |
| Groq | Qwen3-32B | 1.000 | 293ms | 284ms | 366ms | Excellent balance |
| Groq | GPT-OSS-20B | 1.000 | 345ms | 319ms | 438ms | Solid fallback |
| Cerebras | ZAI-GLM-4.7 | 0.989 | 410ms | 375ms | 704ms | Good quality, some variability |

**Recommendation**: Use **Cerebras Qwen3-235B** as the primary model. It delivers perfect quality with the fastest latency in this tier (254ms average). Groq Qwen3-32B makes an excellent fallback.

### Tier 2: Good Quality, Higher Latency
**Criteria**: Quality >= 0.95, Avg Latency < 2000ms

| Provider | Model | Quality | Avg Latency | Median | p95 | Notes |
|---|---|---|---|---|---|---|
| SambaNova | Qwen3-32B | 1.000 | 650ms | 610ms | 850ms | Consistent performance |
| Together | GPT-OSS-20B | 1.000 | 749ms | 647ms | 1113ms | Reliable choice |
| Fireworks | GPT-OSS-20B | 1.000 | 793ms | 770ms | 938ms | Slightly slower than Together |
| Together | Qwen3-235B | 1.000 | 1351ms | 1305ms | 1662ms | Perfect quality, moderate latency |
| DeepInfra | Qwen3-235B | 1.000 | 1483ms | 1316ms | 2337ms | Higher p95 variance |
| Fireworks | GLM-4.7 | 0.989 | 760ms | 743ms | 880ms | Good quality/speed trade-off |
| Fireworks | DeepSeek-V3.1 | 1.000 | 1523ms | 1467ms | 1893ms | Slower but perfect quality |
| Together | Maverick-17B | 0.968 | 966ms | 902ms | 1218ms | Acceptable for non-critical use |

**Use Case**: These models are suitable for scenarios where sub-second latency is acceptable in exchange for perfect or near-perfect quality. Good for batch processing or less interactive contexts.

### Tier 3: Usable with Caveats
**Criteria**: Works but has significant limitations

| Provider | Model | Quality | Avg Latency | Median | p95 | Caveats |
|---|---|---|---|---|---|---|
| OpenAI | GPT-5-nano | 1.000 | 2040ms | 1961ms | 2701ms | Slow, expensive |
| Together | DeepSeek-V3.1 | 0.989 | 1692ms | 1644ms | 2212ms | Near 2s average |
| Groq | Maverick-17B | 0.957 | 456ms | 395ms | 776ms | Quality slightly below threshold |
| OpenRouter | Qwen3-235B | 1.000 | 3552ms | 2791ms | 8074ms | Very high p95 variance |
| Together | Qwen3.5-397B | 1.000 | 4591ms | 3899ms | 7991ms | Requires thinking disabled |
| DeepInfra | DeepSeek-V3.1 | 1.000 | 5092ms | 4827ms | 6692ms | Too slow for interactive use |

**Use Case**: These configurations can work for specific scenarios but are not recommended for general production use due to latency or quality concerns.

### Tier 4: Not Recommended
**Criteria**: Significant quality or reliability issues

| Provider | Model | Quality | Avg Latency | Issue | Impact |
|---|---|---|---|---|---|
| Groq | GPT-OSS-120B | 0.967 | 398ms | ~3% failure rate | Unreliable |
| Cerebras | GPT-OSS-120B | 0.943 | 339ms | ~6% failure rate | High error rate |
| Fireworks | GPT-OSS-120B | 0.950 | 1252ms | ~5% failure rate | Unreliable + slow |
| Anthropic | Claude Haiku 4.5 | 0.978 | 5241ms | Too slow + expensive | Cost prohibitive |
| Fireworks | DeepSeek-V3.2 | 0.150 | 3890ms | Cannot parse JSON | Fundamentally broken |

**Analysis**:
- GPT-OSS-120B has consistent quality issues across all providers (3-6% failure rate)
- Fireworks DeepSeek-V3.2 is completely broken (85% failure rate in JSON parsing)
- Claude Haiku 4.5 works but is too slow and expensive for autocomplete use cases

## Cross-Provider Comparisons

When the same model is available from multiple providers, performance can vary dramatically:

### Qwen3-235B Comparison
| Provider | Avg Latency | Speedup vs Slowest |
|---|---|---|
| Cerebras | 254ms | **14.0x** |
| Together | 1351ms | 2.6x |
| DeepInfra | 1483ms | 2.4x |
| OpenRouter | 3552ms | 1.0x (baseline) |

**Finding**: Cerebras delivers the same model 14x faster than OpenRouter and 5.3x faster than Together.

### GPT-OSS-20B Comparison
| Provider | Avg Latency | Speedup vs Slowest |
|---|---|---|
| Groq | 345ms | **2.3x** |
| Together | 749ms | 1.1x |
| Fireworks | 793ms | 1.0x (baseline) |

**Finding**: Groq is 2.3x faster than Fireworks for the same model.

### DeepSeek-V3.1 Comparison
| Provider | Avg Latency | Speedup vs Slowest |
|---|---|---|
| Fireworks | 1523ms | **3.3x** |
| Together | 1692ms | 3.0x |
| DeepInfra | 5092ms | 1.0x (baseline) |

**Finding**: Fireworks is 3.3x faster than DeepInfra for DeepSeek-V3.1.

## Key Findings

1. **Cerebras Qwen3-235B is the clear winner**: Perfect quality (1.000) at 254ms average latency. This is the best overall configuration tested.

2. **Provider matters as much as model**: The same model can vary by up to 14x in latency depending on the provider. Infrastructure quality is critical.

3. **GPT-OSS-120B has systemic issues**: Despite good latency, this model shows 3-6% failure rates across all providers tested. Not production-ready.

4. **Qwen3.5-397B works but is impractical**: Perfect quality but 4.6s average latency makes it unusable for real-time autocomplete (only available on Together with thinking disabled).

5. **Fireworks DeepSeek-V3.2 is broken**: 85% failure rate in JSON parsing makes this completely unusable. DeepSeek-V3.1 works fine on the same provider.

6. **Groq excels at smaller models**: Qwen3-32B (293ms) and GPT-OSS-20B (345ms) both deliver perfect quality with excellent latency.

7. **Cost vs Performance trade-off**: OpenAI GPT-5-nano delivers perfect quality but at 2040ms and higher API costs, making it impractical for high-frequency autocomplete.

## Recommended Production Configuration

### Primary Configuration
```env
AUTOCOMPLETER_LLM_PROVIDER=cerebras
AUTOCOMPLETER_LLM_MODEL=qwen3-235b
```
- **Quality**: 1.000 (perfect)
- **Latency**: 254ms avg, 256ms median, 289ms p95
- **Reliability**: No observed failures
- **Cost**: Competitive

### Fallback Configuration #1
```env
AUTOCOMPLETER_LLM_PROVIDER=groq
AUTOCOMPLETER_LLM_MODEL=qwen3-32b
```
- **Quality**: 1.000 (perfect)
- **Latency**: 293ms avg, 284ms median, 366ms p95
- **Reliability**: Excellent
- **Cost**: Free tier available

### Fallback Configuration #2
```env
AUTOCOMPLETER_LLM_PROVIDER=groq
AUTOCOMPLETER_LLM_MODEL=gpt-oss-20b
```
- **Quality**: 1.000 (perfect)
- **Latency**: 345ms avg, 319ms median, 438ms p95
- **Reliability**: Excellent
- **Cost**: Free tier available

### Load Balancing Strategy
For production deployments requiring high availability:

1. **Primary**: Cerebras Qwen3-235B (75% of traffic)
2. **Secondary**: Groq Qwen3-32B (20% of traffic)
3. **Tertiary**: SambaNova Qwen3-32B (5% of traffic for testing)

This provides redundancy across three different infrastructure providers while maintaining sub-700ms latency and perfect quality across all tiers.

## Methodology Notes

- **Total runs**: 1008 (v2 + v3 + v4 combined)
- **Configurations tested**: 24 provider/model combinations
- **Scenarios**: Continuation, reply, code completion
- **Quality measurement**: 0-1 scale based on correctness and format compliance
- **Latency measurement**: Time from request to first complete suggestion
- **p95 latency**: 95th percentile, representing worst-case performance for most requests

## Conclusion

The benchmark clearly demonstrates that **Cerebras Qwen3-235B** is the optimal choice for production autocomplete workloads, offering the best combination of quality, latency, and reliability. Groq provides excellent fallback options with both Qwen3-32B and GPT-OSS-20B delivering perfect quality at competitive latencies.

Provider selection matters critically: the same model can perform 14x better on one provider versus another. Infrastructure quality, not just model capabilities, determines real-world performance.
