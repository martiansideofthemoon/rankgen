import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RankGenGenerator():
    def __init__(self, rankgen_encoder, language_model="gpt2-medium", cache_dir=None):
        self.rankgen_encoder = rankgen_encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(language_model, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model, cache_dir=cache_dir)
        self.language_model.to(self.device)
        self.language_model.eval()

    def rankgen_scorer(self, prefix, suffixes, prefix_vector=None):
        rankgen_model = self.rankgen_encoder
        if prefix_vector is None:
            prefix_vector = rankgen_model.encode(prefix, vectors_type="prefix")["embeddings"]
        suffix_vectors = rankgen_model.encode(suffixes, vectors_type="suffix")["embeddings"]
        similarities = torch.matmul(prefix_vector, suffix_vectors.t()).squeeze(dim=0)
        return similarities, prefix_vector, suffix_vectors

    def postprocess(self, outputs):
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_single(self, contexts, temperature=1.0, top_p=0.9, num_samples=10, max_length=115):
        return self.beam_search(contexts=contexts,
                                beam_size=1,
                                temperature=temperature,
                                top_p=top_p,
                                num_tokens=max_length,
                                num_samples=1,
                                max_length=max_length)

    def overgenerate_rerank(self, contexts, temperature=1.0, top_p=0.9, num_samples=10, max_length=115):
        return self.beam_search(contexts=contexts,
                                beam_size=1,
                                temperature=temperature,
                                top_p=top_p,
                                num_tokens=max_length,
                                num_samples=num_samples,
                                max_length=max_length)

    def beam_search(self, contexts, beam_size=2, temperature=1.0, top_p=0.9, num_tokens=20, num_samples=10, max_length=115):
        final_outputs = []
        final_scores = []
        total_generated_tokens = 0
        for ctx in contexts:
            if beam_size == 1 and num_samples == 1:
                prefix_vector = None
            else:
                _, prefix_vector, _ = self.rankgen_scorer(prefix=ctx, suffixes=[ctx])
            beams = [{
                "text": "",
                "eos": False
            } for _ in range(beam_size)]
            while True:
                all_outs = []
                max_new_tokens = min(num_tokens, max_length - total_generated_tokens)
                for beam in beams:
                    # if a beam has ended, add it to all_outs
                    if beam["eos"]:
                        all_outs.append(beam)
                        continue
                    # otherwise generate the next n tokens
                    inputs = self.tokenizer(ctx + beam['text'], truncation=True, padding="longest",
                                            return_tensors="pt", max_length=1024 - max_new_tokens).to(self.device)
                    num_input_tokens = len(inputs['input_ids'][0])
                    with torch.inference_mode():
                        curr_outs = self.language_model.generate(**inputs, do_sample=True, output_scores=True,
                                                                return_dict_in_generate=True,
                                                                max_new_tokens=max_new_tokens, top_k=None, top_p=top_p,
                                                                num_return_sequences=num_samples, temperature=temperature)
                    is_eos = []
                    for curr_out in curr_outs['sequences']:
                        if self.tokenizer.eos_token_id in curr_out:
                            is_eos.append(True)
                        else:
                            is_eos.append(False)
                    curr_outs_text = self.postprocess(curr_outs['sequences'][:, num_input_tokens:])
                    for text, eos in zip(curr_outs_text, is_eos):
                        # update all_outs
                        all_outs.append({
                            "text": beam["text"] + text,
                            "eos": eos
                        })
                # Each beam has total_generated_tokens length
                total_generated_tokens += max_new_tokens
                if len(all_outs) > 1:
                    # skip beam scoring if only one output to choose from
                    scores, _, _ = self.rankgen_scorer(prefix=ctx, suffixes=[x["text"] for x in all_outs], prefix_vector=prefix_vector)
                    top_scores, top_indices = torch.topk(scores, k=beam_size)
                    beams = [all_outs[x] for x in top_indices]  # only track the top k beams
                else:
                    top_scores = torch.Tensor([1.0])
                    top_scores.cuda()
                    beams = all_outs

                for beam in beams:
                    if len(self.tokenizer.tokenize(beam["text"])) >= max_length:
                        beam["eos"] = True

                if all([x["eos"] for x in beams]):
                    final_outputs.append([x["text"] for x in beams])
                    final_scores.append(top_scores)
                    break
        return final_outputs, final_scores