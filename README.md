We introduce CCFC (Core006
& Core–Full–Core), a dual-track, prompt-007
level defense framework designed to mitigate008
LLMs’ vulnerabilities from prompt injection009
and structure-aware jailbreak attacks. CCFC010
operates by first isolating the semantic core of011
a user query via few-shot prompting, and then012
evaluating the query using two complementary013
tracks: a core-only track to ignore adversar-014
ial distractions (e.g., toxic suffixes or prefix015
injections), and a core-full-core (CFC) track016
to disrupt the structural patterns exploited by017
gradient-based or edit-based attacks. The final018
response is selected based on a safety consis-019
tency check across both tracks, ensuring robust-020
ness without compromising on response quality.021
We demonstrate that on both open-source and022
closed-source large language models, CCFC023
consistently drives attack success rates of di-024
verse, strong jailbreak techniques (e.g., DeepIn-025
ception, GCG) down to nearly zero, with only026
a modest runtime overhead and no sacrifice of027
fidelity on benign queries. Our method consis-028
tently outperforms state-of-the-art prompt-level029
defenses, offering a practical and effective so-030
lution for safer LLM deployment.031
