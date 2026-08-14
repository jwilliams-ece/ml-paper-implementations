import numpy as np

from neural_net import CBOW
from data import(
    vocab_size, 
    EMBEDDING_DIM,
    get_data,
)

def main():
    model = CBOW(vocab_size=vocab_size, dimension_size=EMBEDDING_DIM)
    data_pairs, _ , _ = get_data()

    def softmax(logits):
        shifted_logits = logits - np.max(logits)

        exp_logits = np.exp(shifted_logits)
        normalization = exp_logits.sum()

        probabilities = exp_logits / normalization

        return probabilities


    def negative_log_loss(probabilities, target_index):
        target_probability = probabilities[target_index]
        loss = -np.log(target_probability + 1e-10)

        return loss


    def compute_output_gradient(probabilities, target_index):
        output_gradient = probabilities.copy()
        output_gradient[target_index] -= 1

        return output_gradient


    def compute_output_weights_gradient(hidden, output_gradient):
        weights_gradient = np.outer(np.transpose(hidden), output_gradient)

        return weights_gradient


    def compute_hidden_gradient(output_gradient):
        hidden_gradient = output_gradient @ np.transpose(model.output_matrix)

        return hidden_gradient


    def compute_embedding_gradient(context, hidden_gradient):
        context_size = len(context)
        embedding_gradient = hidden_gradient / context_size

        return embedding_gradient


    def update_parameters(learning_rate, context, embedding_gradient,output_weights_gradient):
        model.output_matrix -= learning_rate * output_weights_gradient
        for token_index in context:
            model.embeddings_matrix[token_index] -= (
                learning_rate * embedding_gradient
            )

    def train(epoch):
        lr = 1e-3

        for context, target in data_pairs:
            logits, hidden = model(context)

            # compute softmax and loss
            sfmax_probs = softmax(logits)
            loss = negative_log_loss(sfmax_probs, target)

            do = compute_output_gradient(sfmax_probs,target)

            dW = compute_output_weights_gradient(hidden, do)

            dh = compute_hidden_gradient(do)

            dE = compute_embedding_gradient(context,dh)

            update_parameters(learning_rate=lr,context=context,embedding_gradient=dE,output_weights_gradient=dW)

        if epoch % 1500 == 0:
            print(f"Epoch: {epoch:<15} Loss: {loss}")



    epochs = 100_000
    for epoch in range(epochs):
        train(epoch)
    print("done")



if __name__ == "__main__":
    main()



