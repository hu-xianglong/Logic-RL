1. use git@github.com:camel-ai/loong.git logic domain as training data instead of data/kk/instruct/3ppl/, loong/data/logic/, use a 90/10 split for train and test
2. you need to update reward and evaluation 
3. make this generic, so that later data from other domains can be adopted without too much effort
4. test and validate with a training script with a small dataset sample
Information:
- main_grpo.sh is the entry point for trainning
- eval_kk/eval.sh is the entry point for post-training evaluation
