# Changelog

All notable changes to Agent Diagnostician will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Agent Diagnostician
- Core failure detection framework with 7 detector types
- Support for planning failures (tool use, goal satisfaction, hallucination)
- Support for execution failures (context loss, token exhaustion) 
- Support for termination failures (premature termination, infinite loops)
- Gemini LLM judge integration with API key authentication
- Framework-agnostic trace format supporting multiple tiers of data
- Comprehensive test suite with 115+ real-world test cases
- Evidence-based confidence scoring with human-readable bands
- Graceful degradation for missing trace data
- Embedding-based semantic similarity analysis
- Constraint extraction and validation system
- Multi-stage detection pipeline (rules → embeddings → LLM)

### Performance
- Context loss detection: 83.3% accuracy on validation cases
- Goal satisfaction detection: 100% accuracy on constraint validation
- Tool use detection: 100% accuracy on validation test cases
- Average processing time: < 5 seconds with LLM, < 500ms without

### Documentation
- Complete API reference with examples
- Framework integration guides for LangChain, OpenAI, LlamaIndex, AutoGen
- Contributing guidelines and development setup
- Architecture documentation with design principles
- Performance benchmarking and optimization guides

### Dependencies  
- Python 3.8+ support
- Pydantic v2 for data validation
- sentence-transformers for embedding analysis
- google-generativeai for Gemini LLM integration
- Optional framework integrations (langchain, openai, etc.)

## [0.1.0] - 2024-08-04

### Added
- Initial project structure and core architecture
- BaseDetector interface for failure detection
- AgentTrace and DetectionResult data models
- Basic classifier for aggregating detector results
- Mock LLM judge for development and testing
- Foundation for embedding-based analysis
- Project configuration and development tooling

---

## Development Notes

### Versioning Strategy
- **0.x.x**: Pre-release development versions
- **1.0.0**: First stable release with API guarantees
- **1.x.x**: Feature additions, backwards compatible
- **2.x.x**: Major API changes, migration guides provided

### Release Timeline
- **v0.1.x**: Core architecture and basic detectors (Current)
- **v0.2.x**: Enhanced LLM provider support (OpenAI, Anthropic)
- **v0.3.x**: Advanced framework integrations and performance optimizations
- **v0.4.x**: Real-time streaming analysis and monitoring capabilities  
- **v1.0.0**: Production-ready stable release

### Breaking Changes Policy
We maintain backwards compatibility within major versions. When breaking changes are necessary:
1. Advance notice in release notes (2+ minor versions)
2. Deprecation warnings in code
3. Migration guide in documentation
4. Automated migration tools when possible