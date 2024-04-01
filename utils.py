import sacrebleu


def get_sacrebleu(refs, hyp):
    bleu = sacrebleu.corpus_bleu(hyp, [refs])
    return bleu.score


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None):
    """From fairseq. This returns the label smoothed cross entropy loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        denominator = (1.0 - 1.0*pad_mask)
        denominator = denominator.sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        denominator = 1.0
    
    if ignore_index is not None:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
        
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    loss = loss/denominator
    return loss