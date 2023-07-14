def L1_regularization(model, factor):
  reg_loss = 0
  for param in model.parameters():
    if param.dim() > 1:
      reg_loss += param.norm(1)
  loss = factor * reg_loss

  return loss
