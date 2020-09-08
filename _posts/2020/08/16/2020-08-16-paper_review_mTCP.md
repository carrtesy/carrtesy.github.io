---
title: "Paper Review - mTCP"
date: 2020-08-16
categories:
 - paper review

tags:
 - networking
 - TCP
---

This paper is on mTCP, which implements User-lever TCP Stack.

### Paper Information

Title : mTCP: A Highly Scalable User-level TCP Stack for Multicore Systems

From: 
EunYoung Jeong, Shinae Woo, Muhammad Jamshed, Haewon Jeong
Sunghwan Ihm*, Dongsu Han, and KyoungSoo Park
KAIST *Princeton University

Link: <https://www.usenix.org/conference/nsdi14/technical-sessions/presentation/jeong>

### Abstract & Conclusion

- mTCP: a high-performance user level TCP stack for multicore systems.
	-  translates multiple expensive system calls into a single shared memory reference
	 - allows efficient flow level event aggregation
	 - performs batched packet I/O for high I/O efficiency

- mTCP unleashes the TCP stack from the kernel and directly delivers the benefit of high-performance packet I/O to the transport and application layer
	-  transparent and bi-directional batching of packet  and flow-level events : amortizes the context switch overhead
	- the use of lock-free data structures, cache-aware thread placement, and efficient per-core resource management

- mTCP improves the performance of existing applications by up to 320%. 

### Introduction
- Cause of TCP inefficiency
	- System call overhead
	- Inefficient implementation causing resource contention
	
- Research Goal
	- Multicore scalability of the TCP stack
	- Ease of use ( Application portablity )
	- Ease of deployment ( no kernel modifications )

- Focus
	- TCP in user level -> system call overhead decreases, but processing IPC(inter-process communication) requires context switches
	- This can be solved by batch of packet level and socket level events	

- Contributions
	- Performance gain by packet / socket level batching
	- Integration can be done without requiring significant modification of the kernel
	
	
### Background & Motivation
- Limitations of the Kernel's TCP Stack
	- Lack of connection locality: multiple thread share single socket, and core may be different from running process -> (Megapipe) Local accept queue in each core
	- Shared file descriptor space: lock contention for fd -> (Megapipe) Eliminate layer, and explicitly partition fd space & regular files
 	-  Inefficient per-packet processing -> batch processing
	- System call overhead -> batch processing
	
### Why User-level TCP?
- Lock contention for shared in-kernel data structures, buffer management, and frequent mode switch are main culprits
	- User level TCP stack, with all exisiting optimization techniques, and bring the performance of packe I/O libraries
	
- User level TCP is attractive as:
	 - easily depart from kernel complexity, and can take advantage of high-performance packet I/O library (netmap, DPDK)
	- can apply batch processing (FlexSC, VOS) without extensive kernel modifications
	 - can easily preserve API
	 
### Design

- User-level packet I/O library
	- polling based approach waste CPU cycles, and requires efficient multiplexing between RX/TX -> PacketShader I/O engine (PSIO)

- User-level TCP Stack
	- eliminates many system calls -> separate TCP thread

- Basic TCP Processing
- Lock-free, Per-core Data Structures
	- Thread mapping and flow-level core affinity
	- Multi-core and cache-friendly data structures
	- Efficient TCP timer management
- Batched Event Handling
- Optimizing for Short-lived Connections
	- priority based packet queueing (for control packets)
	- Lightweight Connection Setup

### 