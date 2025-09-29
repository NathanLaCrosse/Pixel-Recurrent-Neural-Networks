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

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True)
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

        # Initialize the first row of hidden vectors - will be updated as it goes along
        previous_row = torch.full((batch_size, pc, self.hidden_size),fill_value=0,dtype=torch.float32)
        current_row = []

        last_col = None
        for row in range(pr):
            # Initialize the previous hidden vector to the left
            last_col = torch.zeros((batch_size, self.hidden_size),dtype=torch.float32) # Initialize last memory column

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

# What a mouthful...
class OmniDirectionalTwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, embedding_size=None, hidden_size=None, num_layers=1, dropout=0):
        super(OmniDirectionalTwoDimensionalGRU, self).__init__()

        if hidden_size is None:
            hidden_size = 2 * input_size

        if embedding_size is not None:
            self.embedding = nn.Linear(in_features=input_size, out_features=embedding_size)
        else:
            embedding_size = input_size

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.grus = nn.ModuleDict({
            "Top-Left" : nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True),
            "Top-Right" : nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True),
            "Bottom-Left" : nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True),
            "Bottom-Right" : nn.GRU(input_size=embedding_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True)
        })

        self.compressors = nn.ModuleDict({
            "Top-Left" : nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            "Top-Right": nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            "Bottom-Left": nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            "Bottom-Right": nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        })

        self.residual_projs = nn.ModuleDict({
            "Top-Left" : nn.Linear(in_features=embedding_size, out_features=self.hidden_size),
            "Top-Right" : nn.Linear(in_features=embedding_size, out_features=self.hidden_size),
            "Bottom-Left" : nn.Linear(in_features=embedding_size, out_features=self.hidden_size),
            "Bottom-Right" : nn.Linear(in_features=embedding_size, out_features=self.hidden_size)
        })

        self.norms = nn.ModuleDict({
            "Top-Left" : nn.LayerNorm(self.hidden_size),
            "Top-Right" : nn.LayerNorm(self.hidden_size),
            "Bottom-Left" : nn.LayerNorm(self.hidden_size),
            "Bottom-Right" : nn.LayerNorm(self.hidden_size)
        })

    def forward(self, x):
        """
                Forward pass of the 2-D GRU

                Args:
                    x: (batch_size, patch_rows, patch_cols, block_dim**2). Blocks are flattened across final dimension

                """

        # The sequence length is known: im_rows * im_cols
        batch_size, pr, pc, bd = x.size()

        # Storage for previous layer's output
        output = {
            "Top-Left" : torch.zeros_like(x),
            "Top-Right": torch.zeros_like(x),
            "Bottom-Left": torch.zeros_like(x),
            "Bottom-Right": torch.zeros_like(x)
        }

        for layer in range(self.num_layers):
            # Initialize previous hidden data
            prev_rows = {
                "Top-Left" : torch.full((batch_size, pc, self.hidden_size), fill_value=0, dtype=torch.float32),
                "Top-Right" : torch.full((batch_size, pc, self.hidden_size), fill_value=0, dtype=torch.float32),
                "Bottom-Left" : torch.full((batch_size, pc, self.hidden_size), fill_value=0, dtype=torch.float32),
                "Bottom-Right" :torch.full((batch_size, pc, self.hidden_size), fill_value=0, dtype=torch.float32)
            }

            # Initialize current hidden rows
            cur_rows = {
                "Top-Left": [],
                "Top-Right": [],
                "Bottom-Left": [],
                "Bottom-Right": []
            }

            # Store the direction of each of the GRUs in (row, col) format
            directions = {
                "Top-Left" : (1, 1),
                "Top-Right" : (1, -1),
                "Bottom-Left" : (-1, 1),
                "Bottom-Right" : (-1, -1)
            }
            # Store the starting locations of each of the GRUs in (row, col) format
            starts = {
                "Top-Left" : (0, 0),
                "Top-Right" : (0, pc-1),
                "Bottom-Left" : (pr-1, 0),
                "Bottom-Right" : (pr-1, pc-1)
            }

            last_col = None
            for row in range(pr):
                # Initialize previous hidden state to the (left/right)
                last_col = {
                    "Top-Left" : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32),
                    "Top-Right" : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32),
                    "Bottom-Left" : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32),
                    "Bottom-Right" : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32)
                }

                for col in range(pc):
                    # For each GRU, calculate its forward step
                    for key in self.grus.keys():
                        # Grab the above/below row hidden vector
                        last_row = prev_rows[key][:, starts[key][1] + directions[key][1] * col, :]

                        # Load in the x_ij
                        x_ij = x[:, starts[key][0] + directions[key][0] * row, starts[key][1] + directions[key][1] * col, :]
                        x_ij = self.embedding(x_ij)

                        # Combine the two previous hidden vectors (one from above/below one from left/right)
                        previous_data = torch.cat((last_row, last_col[key]), dim=1)

                        # Reshaping
                        x_ij = x_ij.view(batch_size, 1, self.embedding_size)
                        previous_data = previous_data.view(1, batch_size, self.hidden_size * 2)

                        # Throw data into the underlying gru
                        o_ij, h_ij = self.grus[key](x_ij, previous_data)
                        h_ij = h_ij[-1]

                        if self.dropout > 0:
                            h_ij = F.dropout(h_ij, p=self.dropout)

                        # Translate new hidden vector back into the proper size
                        h_ij = self.compressors[key](h_ij)

                        # Create a residual based off of x_ij
                        residual = self.residual_projs[key](x_ij).view(batch_size, self.hidden_size)
                        h_ij = residual + h_ij

                        h_ij = self.norms[key](h_ij)

                        # Update current row values
                        cur_rows[key].append(h_ij)
                        last_col[key] = h_ij

                # Update previous row data now that the row is finished
                for key in self.grus.keys():
                    saved_row = cur_rows[key]

                    # Make sure its saved in the correct direction..?
                    if directions[key][1] == -1:
                        # Reverse the column order for right-to-left processing
                        saved_row = saved_row[::-1]

                    prev_rows[key] = torch.stack(saved_row, dim=1)
                    cur_rows[key] = []

        # Package up the final hidden vectors
        h_final = torch.cat((last_col["Top-Left"], last_col["Top-Right"], last_col["Bottom-Left"], last_col["Bottom-Right"]), dim=1)
        """, last_col["Bottom-Left"], last_col["Bottom-Right"]"""
        return h_final


# # Update previous hidden values
#                 current_row.append(h_ij)
#                 last_col = h_ij
#
#             # Update previous row
#             current_row = torch.stack(current_row, dim=1) # Now should be (batch_size, seq_len, hidden_size)
#             previous_row = current_row
#             current_row = []

# def get_idx(starts, directions, key, i):
#     return starts[key] + directions[key][1] * i

if __name__ == '__main__':
    net = OmniDirectionalTwoDimensionalGRU(9, 10, 12, 1)

    test_tensor = torch.rand((4, 2, 3, 9))
    test_result = net(test_tensor)

    labels = torch.rand((4,1))
    args = torch.argmax(test_result)

    print("hi")
