import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import embedding


# What a mouthful...
class UnderlyingTwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, embedding_size=None, hidden_size=None, dropout=0, omnidirectionality=False):
        super(UnderlyingTwoDimensionalGRU, self).__init__()
        if not omnidirectionality:
            self.direction_ref = ["Top-Left"]
        else:
            self.direction_ref = ["Top-Left","Top-Right","Bottom-Left","Bottom-Right"]

        if hidden_size is None:
            hidden_size = 2 * input_size

        if embedding_size is not None:
            self.embedding = nn.Linear(in_features=input_size, out_features=embedding_size)
        else:
            embedding_size = input_size # Make sure GRUs have correct input size
            self.embedding = None

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout

        self.grus = nn.ModuleDict({ref : nn.GRU(input_size=embedding_size,
            hidden_size=2*hidden_size, num_layers=1, batch_first=True) for ref in self.direction_ref})

        self.compressors = nn.ModuleDict({ref: nn.Linear(in_features=2*hidden_size,
            out_features=hidden_size) for ref in self.direction_ref})

        self.residual_projs = nn.ModuleDict({ref : nn.Linear(in_features=embedding_size,
            out_features=self.hidden_size) for ref in self.direction_ref})

        self.norms = nn.ModuleDict({ref : nn.LayerNorm(self.hidden_size) for ref in self.direction_ref})

    def forward(self, x, return_final_row_col=False, preset_row=None, preset_col=None):
        """
        Forward pass of the 2-D GRU

        Args:
            x: (batch_size, patch_rows, patch_cols, block_dim**2). Blocks are flattened across final dimension
                - for unidirectional
            x: (D, batchsize, patch_rows, patch_cols, block_dim**2). Format for multilayered input (from previous layer)
                - D is either 1 or 4 depending on directionality

        Returns:
            output: (D, batch_size, patch_rows, patch_cols, block_dim**2). The hidden value tensor that stores all 
                hidden vectors calculated by the MultidimensionalGRU. D = 1 if omnidirectionality = False, 4 otherwise
            h_final: (batch_size, hidden_size * D). The concatenation of all final hidden vectors of the RNN.
                

        """

        # The sequence length is known: im_rows * im_cols
        mode = "first_layer"
        if len(x.size()) == 4:
            batch_size, pr, pc, bd = x.size()
        else:
            mode = "extra_layer"
            _, batch_size, pr, pc, bd = x.size()

        # Storage for previous layer's output
        output = {ref : torch.zeros((batch_size, pr, pc, self.hidden_size))
            for ref in self.direction_ref}

        # Initialize previous hidden data
        if preset_row is None:
            prev_rows = {ref : torch.full((batch_size, pc, self.hidden_size), fill_value=0,
                dtype=torch.float32) for ref in self.direction_ref}
        else:
            prev_rows = preset_row

        # Initialize current hidden rows
        cur_rows = {ref: [] for ref in self.direction_ref}

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

        # Initialize - if requested, final row and col data
        if return_final_row_col:
            final_row = {ref : torch.zeros(batch_size, pc, self.hidden_size) for ref in self.direction_ref}
            final_col = {ref : torch.zeros(batch_size, pr, self.hidden_size) for ref in self.direction_ref}
        else:
            final_row = None
            final_col = None

        last_col = None
        for row in range(pr):
            # Initialize previous hidden state to the (left/right)
            if preset_col is None:
                last_col = {ref : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32) for ref in self.direction_ref}
            else:
                last_col = {ref: preset_col[ref][:, row, :] for ref in self.direction_ref}

            for col in range(pc):
                # For each GRU, calculate its forward step
                for key in self.direction_ref:
                    # Grab the above/below row hidden vector
                    last_row = prev_rows[key][:, starts[key][1] + directions[key][1] * col, :]

                    # Load in the x_ij
                    if mode == "first_layer":
                        x_ij = x[:, starts[key][0] + directions[key][0] * row, starts[key][1] + directions[key][1] * col, :]
                    else:
                        x_ij = x[self.direction_ref.index(key), :, starts[key][0] + directions[key][0] * row, starts[key][1] + directions[key][1] * col, :]

                    if self.embedding is not None:
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

                    if return_final_row_col and col == pc - 1:
                        final_col[key][:, starts[key][0] + directions[key][0] * row, :] = last_col[key]

            # Update previous row data now that the row is finished
            for key in self.grus.keys():
                saved_row = cur_rows[key]

                # Make sure its saved in the correct direction..?
                if directions[key][1] == -1:
                    # Reverse the column order for right-to-left processing
                    saved_row = saved_row[::-1]
                saved_row = torch.stack(saved_row, dim=1)

                row_dex = starts[key][0] + directions[key][0] * row
                output[key][:, row_dex, :, :] = saved_row

                prev_rows[key] = saved_row
                cur_rows[key] = []

                if return_final_row_col:
                    final_row[key] = saved_row

        # Package up the final hidden vectors
        h_final = torch.cat([last_col[key] for key in self.direction_ref], dim=1)
        output = torch.stack([output[key] for key in self.direction_ref])
        """, last_col["Bottom-Left"], last_col["Bottom-Right"]"""
        if not return_final_row_col:
            return output, h_final
        else:
            return output, h_final, final_row, final_col



class TwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size=None, omnidirectionality=False):
        super(TwoDimensionalGRU, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers

        # input_size, embedding_size=None, hidden_size=None, dropout=0, omnidirectionality=False
        self.inGRU = UnderlyingTwoDimensionalGRU(input_size, embedding_size, hidden_size=hidden_size, omnidirectionality=omnidirectionality)
        self.deeper_layers = nn.ModuleList([
            UnderlyingTwoDimensionalGRU(hidden_size, embedding_size=None, hidden_size=hidden_size,
                omnidirectionality=omnidirectionality) for i in range(self.num_layers - 1)
        ])

    def forward(self, x):
        output, h_final = self.inGRU(x)
        for layer in self.deeper_layers:
            output, h_final = layer(output)
        return output, h_final


if __name__ == '__main__':
    # net = OmniDirectionalTwoDimensionalGRU(9, 10, 12, omnidirectionality=False)

    # net = TwoDimensionalGRU(9, 12, 2, embedding_size=10, omnidirectionality=True)
    net = UnderlyingTwoDimensionalGRU(9, 10, 12,  omnidirectionality=False)

    test_tensor = torch.rand((4, 2, 3, 9))
    # output, test_result, final_row, final_col = net.forward(test_tensor, return_final_row_col=True)
    #
    # print("Verifying output row:")
    # print(output[0, 0, 1, :, :])
    # print(final_row["Top-Left"][0])
    #
    # print("Verifying output col:")
    # print(output[0, 0, :, 2, :])
    # print(final_col["Top-Left"][0])

    initial_hiddens = {"Top-Left" : torch.ones(4, 3, 12)}
    output, test_result, final_row, final_col = net.forward(test_tensor, return_final_row_col=True, preset_col=initial_hiddens)

    # labels = torch.rand((4,1))
    # args = torch.argmax(test_result)

    print("hi")
