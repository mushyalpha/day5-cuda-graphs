# Day 5/45 of AI Inference Engineering: CUDA Graphs

Today I used CUDA graphs to double the serving speed of my custom "baby-vLLM" engine for a single user. 
But when I scaled up to 64 users, I accidentally uncovered a hidden PyTorch bug copying 105 GB of memory for no reason. 🧵👇

**Some context:** 
Earlier this week, I built a custom "baby-vLLM" engine with the goal of getting a first-principled understanding of why vLLM is powerful. 
I intentionally started bare-bones so I could add the optimizations myself, layer by layer. 

**The Day 4 Problem:**
Yesterday, I hit my first major bottleneck. Even with my custom engine, the GPU was spending 60% of its time doing absolutely nothing. 
Why? Because Python was sending 1,951 tiny, separate instructions to the GPU for every single word generated. 
The GPU is so incredibly fast that it would finish the math in a microsecond, and then sit around waiting for Python to send the next instruction. 
Out of a 21.2 ms step, the H100 spent 12.7 ms just staring at the wall. You don't rent a $30,000 GPU to wait on Python.

**The Day 5 Solution:**
Today I applied CUDA graphs to my custom-built engine to solve this. Instead of sending 1,951 separate instructions, I locked all the memory into static buffers and captured the whole decode step into a single CUDA graph.
It worked miraculously. 
This slashed the total time down to 9.5 ms. The idle waiting time dropped from 12.7 ms to just 1.2 ms, and the serving speed more than doubled from 47 tokens/sec to 105 tokens/sec.

The absolute physical memory bandwidth limit of the H100 for a 7B model is 4.5 ms. By dropping my step time to 9.5 ms, I am now operating at nearly 50% of the theoretical hardware limit.

**But surprisingly...**
When I simulated a heavy production load (Batch 64, or 64 concurrent users), my shiny new engine completely choked. 
It took 187 ms per step. That is 8x slower than the basic, unoptimized HuggingFace code I started with on Day 4. What happened?

I profiled it and found a catastrophe. 
A specific PyTorch attention setting (`enable_gqa=True`) was secretly triggering a fallback that needlessly duplicated memory. PyTorch was blindly copying 105 Gigabytes of data back and forth inside the GPU on every single step.

I applied a 2-line fix. By manually "folding" the tensor shape, I completely bypassed PyTorch's memory expansion. 
The latency instantly plummeted from a disastrous 187 ms down to an incredible 20.9 ms. I fixed an 8x slowdown with a reshape.

**So for Day 6:**
CUDA graphs are done. We've squeezed everything we can out of standard PyTorch. 
Tomorrow, I leave Python behind. I'll be writing a custom Triton kernel to handle Paged Attention natively on the GPU hardware.
