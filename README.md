# Day 5/45 of AI Inference Engineering

Today I used CUDA graphs to double the serving speed of my small vLLM engine for a single user.

**Some context:** Earlier this week, I built a custom "baby-vLLM" engine with the goal of understanding from first principles why vLLM is so fast. I intentionally left out heavy optimisations so I could understand them by adding them one at a time as I go.

In my last post, I profiled PyTorch and found that it wastes 60% of its time just waiting for CPU to launch kernels. 

Why? Because Python was sending 1 951 separate instructions to the GPU for every single token generated. A single step was taking 21 ms of which 12.7 ms the GPU was doing nothing.

Today I applied CUDA graphs to my custom built engine solve this. Instead of sending 1 951 separate instructions, I locked all the memory into static buffers and captured the whole decode step into a single CUDA graph.   

This slashed the time down to 9.5 ms. The idle waiting time dropped from 12.7 ms to 1.2 ms. And the serving speed more than doubled from 47 tokens/sec to 105 tokens/sec

The absolute physical memory bandwidth limit of the H100 for a 7B model is 4.5 ms. By dropping my step time to 9.5 ms, I am now operating at roughly 50% of the theoretical hardware limit.

Tomorrow I stop using PyTorch's attention entirely.
I will write my own paged attention kernel in Triton.
