# NPU Whisper Encoder Documentation Index

**Project**: Whisper Encoder NPU Optimization
**Target Hardware**: AMD XDNA1 (Phoenix) and XDNA2 (Strix)
**Goal**: 95% code reuse, 1.8-2x performance scaling
**Last Updated**: November 17, 2025

---

## Quick Navigation

### For New Developers
👉 **Start Here**: [QUICK_START_XDNA1_XDNA2.md](./QUICK_START_XDNA1_XDNA2.md)

### For Project Managers
👉 **Executive Overview**: [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md)

### For Architects
👉 **Technical Deep Dive**: [XDNA1_XDNA2_ARCHITECTURE.md](./XDNA1_XDNA2_ARCHITECTURE.md)

### For Performance Analysis
👉 **Benchmarks**: [KERNEL_COMPARISON_XDNA1_XDNA2.md](./KERNEL_COMPARISON_XDNA1_XDNA2.md)

### For Code Reviews
👉 **Quality Guidelines**: [PORTABILITY_CHECKLIST.md](./PORTABILITY_CHECKLIST.md)

---

## Documentation Suite

### 1. [XDNA1_XDNA2_ARCHITECTURE.md](./XDNA1_XDNA2_ARCHITECTURE.md)
**Technical architecture and separation strategy**

**Topics Covered**:
- Hardware architecture comparison (XDNA1 vs XDNA2)
- Directory structure and organization
- Separation strategy (what to share, what to separate)
- Portability approach and principles
- API design for cross-platform compatibility
- Performance scaling analysis
- Migration path from XDNA1 to XDNA2

**Who Should Read**: Technical leads, architects, senior developers

**Key Takeaways**:
- 95% code reuse is achievable
- XDNA2 has 2x columns (4→8), 2x bandwidth (25.6→51.2 GB/s)
- C++ kernels are 100% portable
- MLIR handles all hardware differences

**Time to Read**: 30-40 minutes

---

### 2. [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md)
**4-phase integration timeline and milestones**

**Topics Covered**:
- **Phase 1**: Copy optimized kernels (1 week)
- **Phase 2**: IRON API migration (1 week)
- **Phase 3**: Multi-column optimization (1 week)
- **Phase 4**: XDNA2 preparation (2 weeks)
- Success criteria for each phase
- Risk mitigation strategies
- Timeline and resource estimates

**Who Should Read**: Project managers, team leads, stakeholders

**Key Takeaways**:
- 5-week timeline to XDNA2-ready codebase
- Incremental value delivery each week
- XDNA1 optimization happens in parallel with XDNA2 prep
- Clear milestones and deliverables

**Time to Read**: 25-35 minutes

---

### 3. [KERNEL_COMPARISON_XDNA1_XDNA2.md](./KERNEL_COMPARISON_XDNA1_XDNA2.md)
**Performance comparison and scaling analysis**

**Topics Covered**:
- Kernel-by-kernel performance comparison
- XDNA1 current state (4 columns)
- XDNA2 projected performance (8 columns)
- Scaling factors (1.7-2.0x expected)
- Full encoder layer analysis
- Portability notes for each kernel
- Performance recommendations

**Who Should Read**: Performance engineers, optimization specialists, PMs

**Key Takeaways**:
- MatMul: 1.84x speedup on XDNA2
- Attention: 1.80x speedup on XDNA2
- Full encoder layer: 1.81x speedup (15ms → 8.3ms)
- 95-99% code reuse per kernel

**Time to Read**: 20-30 minutes

---

### 4. [QUICK_START_XDNA1_XDNA2.md](./QUICK_START_XDNA1_XDNA2.md)
**Developer quick start guide**

**Topics Covered**:
- 5-minute quick start (run first kernel)
- Understanding the codebase structure
- Common development tasks
  - Compile a kernel
  - Test a kernel
  - Add a new kernel
  - Port kernel to XDNA2
- Development workflow
- Testing guide
- Troubleshooting common issues
- Best practices
- Useful commands reference

**Who Should Read**: New developers, contributors

**Key Takeaways**:
- Get running in 5 minutes
- Clear task-based instructions
- Practical code examples
- Troubleshooting for common problems

**Time to Read**: 15-25 minutes (reference as needed)

---

### 5. [PHASE1_XDNA2_INTEGRATION_ADDENDUM.md](./PHASE1_XDNA2_INTEGRATION_ADDENDUM.md)
**How XDNA2 work enhances Phase 1**

**Topics Covered**:
- How XDNA2 preparation helps Phase 1 goals
- Updated performance targets
- Quick wins from XDNA2 work
- Updated Phase 1 timeline
- Addressing concerns about scope
- Updated deliverables

**Who Should Read**: Phase 1 team members, stakeholders, PM

**Key Takeaways**:
- XDNA2 prep doesn't slow down Phase 1
- Better code organization makes optimization easier
- IRON API helps XDNA1 performance too
- Same 5-week timeline, better results

**Time to Read**: 15-20 minutes

---

### 6. [PORTABILITY_CHECKLIST.md](./PORTABILITY_CHECKLIST.md)
**Code review guidelines and quality checklist**

**Topics Covered**:
- Quick checklist (C++, MLIR, Python, Testing)
- Detailed checklists for each component
- File location and naming conventions
- Code review checklist
- Common mistakes and how to fix them
- Portability validation steps
- Quick reference guide

**Who Should Read**: All developers, code reviewers

**Key Takeaways**:
- Clear guidelines for portable code
- What goes where (directory structure)
- Common mistakes to avoid
- Review checklist before submitting

**Time to Read**: 10-20 minutes (reference as needed)

---

## Documentation Map

```
                    ┌─────────────────────────────────┐
                    │  START: docs/README.md (YOU)   │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────┴───────────────┐
                    │         Who are you?            │
                    └┬────────────┬───────────┬───────┘
                     │            │           │
                     │            │           │
              ┌──────▼──────┐ ┌──▼────┐  ┌──▼─────────┐
              │ New Dev     │ │  PM   │  │ Architect  │
              └──────┬──────┘ └──┬────┘  └──┬─────────┘
                     │            │          │
                     │            │          │
              ┌──────▼──────────┐ │   ┌──────▼──────────────┐
              │ QUICK_START     │ │   │ ARCHITECTURE        │
              └──────┬──────────┘ │   └──────┬──────────────┘
                     │            │          │
                     │     ┌──────▼────────┐ │
                     │     │  ROADMAP      │ │
                     │     └──────┬────────┘ │
                     │            │          │
                     │     ┌──────▼────────┐ │
                     │     │  COMPARISON   │ │
                     │     └──────┬────────┘ │
                     │            │          │
              ┌──────▼────────────▼──────────▼──────┐
              │  PORTABILITY_CHECKLIST              │
              │  (Everyone uses this)               │
              └─────────────────────────────────────┘
```

---

## Additional Documentation

### In Parent Directory (`../`)

**Phase 1 Documentation**:
- `README_PHASE1.md` - Phase 1 overview
- `PHASE1_DAY1_EXECUTIVE_SUMMARY.md` - Day 1 results
- `PHASE1_PROGRESS.md` - Daily progress log
- `PHASE1_QUICK_REFERENCE.md` - Quick reference

**Kernel Status**:
- `KERNEL_STATUS.md` - Current kernel status
- `ATTENTION_VALIDATION_RESULTS.md` - Attention testing
- `MATMUL_BATCHING_ANALYSIS.md` - MatMul optimization

**Technical Reports**:
- `NPU_MATMUL_PERFORMANCE_ANALYSIS.md` - MatMul profiling
- `ATTENTION_ACCURACY_FINDINGS.md` - Attention validation
- `GELU_IMPLEMENTATION.md` - GELU kernel details

### Online Resources

**AMD Documentation**:
- MLIR-AIE: https://github.com/Xilinx/mlir-aie
- XRT: https://xilinx.github.io/XRT/
- Peano Compiler: AMD internal documentation

**Project Repositories**:
- Main: https://github.com/Unicorn-Commander/Unicorn-Amanuensis
- NPU Core: https://github.com/Unicorn-Commander/unicorn-npu-core

---

## Reading Paths

### Path 1: "I want to get started quickly"

1. [QUICK_START_XDNA1_XDNA2.md](./QUICK_START_XDNA1_XDNA2.md) (15 min)
   - Run your first kernel
   - Understand directory structure
   - Try common tasks

2. [PORTABILITY_CHECKLIST.md](./PORTABILITY_CHECKLIST.md) (10 min)
   - Learn what goes where
   - Review guidelines

3. Start coding! Reference other docs as needed

**Total Time**: 25 minutes to productivity

---

### Path 2: "I need to understand the architecture"

1. [XDNA1_XDNA2_ARCHITECTURE.md](./XDNA1_XDNA2_ARCHITECTURE.md) (35 min)
   - Deep dive into hardware
   - Separation strategy
   - Portability approach

2. [KERNEL_COMPARISON_XDNA1_XDNA2.md](./KERNEL_COMPARISON_XDNA1_XDNA2.md) (25 min)
   - Performance analysis
   - Scaling expectations

3. [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md) (30 min)
   - Implementation timeline
   - Milestones

**Total Time**: 90 minutes for comprehensive understanding

---

### Path 3: "I'm the PM and need an executive summary"

1. [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md) (30 min)
   - Timeline and phases
   - Resource requirements
   - Success criteria

2. [KERNEL_COMPARISON_XDNA1_XDNA2.md](./KERNEL_COMPARISON_XDNA1_XDNA2.md) (20 min)
   - Expected performance gains
   - ROI analysis

3. [PHASE1_XDNA2_INTEGRATION_ADDENDUM.md](./PHASE1_XDNA2_INTEGRATION_ADDENDUM.md) (15 min)
   - How XDNA2 helps Phase 1
   - Risk mitigation

**Total Time**: 65 minutes for decision-making

---

### Path 4: "I'm reviewing code"

1. [PORTABILITY_CHECKLIST.md](./PORTABILITY_CHECKLIST.md) (15 min)
   - Review guidelines
   - Common mistakes

2. [QUICK_START_XDNA1_XDNA2.md](./QUICK_START_XDNA1_XDNA2.md) - Best Practices section (5 min)
   - Coding standards

3. Use checklists during review

**Total Time**: 20 minutes prep, 5-10 minutes per review

---

## Document Status

| Document | Status | Last Updated | Next Review |
|----------|--------|--------------|-------------|
| XDNA1_XDNA2_ARCHITECTURE.md | ✅ Complete | Nov 17, 2025 | After Phase 2 |
| XDNA2_INTEGRATION_ROADMAP.md | ✅ Complete | Nov 17, 2025 | After Phase 1 |
| KERNEL_COMPARISON_XDNA1_XDNA2.md | ✅ Complete | Nov 17, 2025 | After multi-column |
| QUICK_START_XDNA1_XDNA2.md | ✅ Complete | Nov 17, 2025 | After Phase 1 |
| PHASE1_XDNA2_INTEGRATION_ADDENDUM.md | ✅ Complete | Nov 17, 2025 | After Phase 1 |
| PORTABILITY_CHECKLIST.md | ✅ Complete | Nov 17, 2025 | After first XDNA2 port |
| README.md | ✅ Complete | Nov 17, 2025 | Quarterly |

---

## Contribution Guidelines

### Adding Documentation

1. Create markdown file in `docs/`
2. Follow naming convention: `TOPIC_NAME.md`
3. Update this README.md index
4. Include:
   - Date and version
   - Purpose statement
   - Table of contents
   - Clear sections
   - Examples where applicable

### Updating Documentation

1. Update document
2. Update "Last Updated" date
3. Update version number (if major changes)
4. Note changes in document history section

### Documentation Standards

- **Format**: Markdown (`.md`)
- **Line Length**: No hard limit (use natural paragraphs)
- **Code Blocks**: Use syntax highlighting (```python, ```c, ```mlir, ```bash)
- **Headings**: Use ATX style (`#`, `##`, `###`)
- **Tables**: Use GitHub-flavored markdown tables
- **Links**: Use relative paths for internal docs

---

## Glossary

**XDNA1**: AMD's 1st generation NPU (Phoenix, 4 columns, 16 TOPS INT8)
**XDNA2**: AMD's 2nd generation NPU (Strix, 8 columns, 32+ TOPS INT8)
**MLIR**: Multi-Level Intermediate Representation (compiler framework)
**IRON**: Integrated Runtime for Orchestrating NPUs (modern MLIR API)
**XRT**: Xilinx Runtime (driver for NPU)
**XCLBIN**: Xilinx Binary (compiled NPU kernel)
**AIE**: AI Engine (NPU compute core)
**ObjectFIFO**: IRON API for data movement
**DMA**: Direct Memory Access
**Shim Tile**: Interface tile (row 0)
**Compute Tile**: Processing tile (rows 2-4)
**Memory Tile**: L2 cache tile (rows 0-1)

---

## Support

### Questions?

- **Technical**: Review [QUICK_START_XDNA1_XDNA2.md](./QUICK_START_XDNA1_XDNA2.md) troubleshooting section
- **Architecture**: See [XDNA1_XDNA2_ARCHITECTURE.md](./XDNA1_XDNA2_ARCHITECTURE.md)
- **Process**: Check [XDNA2_INTEGRATION_ROADMAP.md](./XDNA2_INTEGRATION_ROADMAP.md)

### Contributing

1. Read relevant documentation first
2. Follow [PORTABILITY_CHECKLIST.md](./PORTABILITY_CHECKLIST.md)
3. Update docs with your changes
4. Submit for review

---

## Summary

This documentation suite provides comprehensive coverage of the XDNA1/XDNA2 NPU optimization project:

- ✅ **Architecture**: Deep technical understanding
- ✅ **Roadmap**: Clear timeline and milestones
- ✅ **Comparison**: Performance analysis
- ✅ **Quick Start**: Fast onboarding
- ✅ **Integration**: Phase 1 alignment
- ✅ **Quality**: Portability guidelines

**Total Documentation**: 6 comprehensive guides covering:
- 50+ pages of technical content
- Code examples and templates
- Performance benchmarks
- Best practices
- Troubleshooting guides
- Review checklists

**Goal**: Enable team to achieve 95% code reuse between XDNA1 and XDNA2 with 1.8-2x performance scaling.

---

**Index Version**: 1.0
**Last Updated**: November 17, 2025
**Maintained By**: NPU Documentation Team
**Questions**: See individual document authors
