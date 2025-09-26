import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import embedding


class TwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, embedding_size=None, hidden_size=None, num_layers=1, dropout=0):
        # Note: internal gru actually has 2*hidden_size, which is later truncated
        # Hidden size is actually the size of the outputs...

        super(TwoDimensionalGRU, self).__init__()

        if hidden_size is None:
            hidden_size = 2*input_size

        if embedding_size is not None:
            self.embedding = nn.Linear(in_features=input_size, out_features=embedding_size)
        else:
            embedding_size = input_size

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.compress = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)

    def forward(self, x : torch.Tensor):
        """
        Forward pass of the 2-D GRU

        Args:
            x: (batch_size, patch_rows, patch_cols, block_dim**2). Blocks are flattened across final dimension

        """

        # The sequence length is known: im_rows * im_cols
        batch_size, pr, pc, bd = x.size()
        seq_len = pr * pc

        x = x.flatten(start_dim=1,end_dim=2) # (batch_size, seq_len, block_dim)

        # Initialize the first row of memory - will be updated as it goes along
        previous_row = torch.full((batch_size, pc, self.hidden_size),fill_value=0,dtype=torch.float32)
        current_row = []

        last_col = None
        for row in range(pr):
            last_col = torch.zeros((batch_size, self.hidden_size)) # Initialize last memory column

            for col in range(pc):
                last_row = previous_row[:, col, :] # Now is (batch_size, hidden_size)

                x_in = x[:,row*pc + col,:] # Load our x_(i,j)

                # Embed x
                x_in = self.embedding(x_in)

                previous_data = torch.cat((last_row, last_col), dim=1)

                # Reshaping shenangins
                x_in = x_in.view(batch_size, 1, self.embedding_size)
                previous_data = previous_data.view(1, batch_size, self.hidden_size * 2)

                o_ij, h_ij = self.gru(x_in, previous_data)
                h_ij = h_ij[-1]

                if self.dropout > 0:
                    h_ij = F.dropout(h_ij, p=self.dropout)

                # Translate new hidden vector back into the proper size
                h_ij = self.compress(h_ij)

                # Update previous hidden values
                current_row.append(h_ij)
                last_col = h_ij

            # Update previous row
            current_row = torch.stack(current_row, dim=1) # Now should be (batch_size, seq_len, hidden_size)
            previous_row = current_row
            current_row = []

        return last_col

if __name__ == '__main__':
    net = TwoDimensionalGRU(9, 10, 12, 1)

    test_tensor = torch.rand((3, 2, 2, 9))
    test_result = net(test_tensor)

    print("hi")