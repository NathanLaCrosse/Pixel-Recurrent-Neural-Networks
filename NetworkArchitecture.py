import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import embedding


# What a mouthful...
class UnderlyingTwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, embedding_size=None, hidden_size=None, dropout=0, omnidirectionality=False, device=torch.device('cpu')):
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
        self.device = device

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
        output = {ref : torch.zeros((batch_size, pr, pc, self.hidden_size), device=self.device)
            for ref in self.direction_ref}

        # Initialize previous hidden data
        if preset_row is None:
            prev_rows = {ref : torch.full((batch_size, pc, self.hidden_size), fill_value=0,
                dtype=torch.float32, device=self.device) for ref in self.direction_ref}
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
            final_row = {ref : torch.zeros(batch_size, pc, self.hidden_size, device=self.device) for ref in self.direction_ref}
            final_col = {ref : torch.zeros(batch_size, pr, self.hidden_size, device=self.device) for ref in self.direction_ref}
        else:
            final_row = None
            final_col = None

        last_col = None
        for row in range(pr):
            # Initialize previous hidden state to the (left/right)
            if preset_col is None:
                last_col = {ref : torch.zeros((batch_size, self.hidden_size),dtype=torch.float32, device=self.device) for ref in self.direction_ref}
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

                    # Main loop is in this single step method
                    h_ij = self.single_step(x_ij, last_row, last_col[key], batch_size, key)

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

    def single_step(self, x, hidden_vert, hidden_horiz, batch_size, key):
        """
        Defines a single step of the GRU network, given a vertical hidden and a horizontal hidden.
        Useful for redefining the forward method.
        """

        if self.embedding is not None:
            x = self.embedding(x)

        # Combine the two previous hidden vectors (one from above/below one from left/right)
        previous_data = torch.cat((hidden_vert, hidden_horiz), dim=1)

        # Reshaping
        x = x.view(batch_size, 1, self.embedding_size)
        previous_data = previous_data.view(1, batch_size, self.hidden_size * 2)

        # Throw data into the underlying gru
        o, h = self.grus[key](x, previous_data)
        h = h[-1]

        # Translate new hidden vector back into the proper size
        h = self.compressors[key](h)

        # Create a residual based off of x_ij
        residual = self.residual_projs[key](x).view(batch_size, self.hidden_size)
        h = residual + h

        h = self.norms[key](h)

        return h

class TwoDimensionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size=None,
                 omnidirectionality=False, device=torch.device('cpu')):
        super(TwoDimensionalGRU, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers

        # input_size, embedding_size=None, hidden_size=None, dropout=0, omnidirectionality=False
        self.inGRU = UnderlyingTwoDimensionalGRU(input_size, embedding_size, hidden_size=hidden_size,
                omnidirectionality=omnidirectionality, device=device)
        self.deeper_layers = nn.ModuleList([
            UnderlyingTwoDimensionalGRU(hidden_size, embedding_size=None, hidden_size=hidden_size,
                omnidirectionality=omnidirectionality, device=device) for i in range(self.num_layers - 1)
        ])

        self.device = device

    def forward(self, x):
        output, h_final = self.inGRU(x)
        for layer in self.deeper_layers:
            output, h_final = layer(output)
        return output, h_final

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, omnidirectionality, class_count):
        super(MNISTClassifier, self).__init__()

        self.recurrent = TwoDimensionalGRU(input_size = input_size, embedding_size = embedding_size,
                                              hidden_size = hidden_size, num_layers = num_layers,
                                              omnidirectionality = omnidirectionality)
        self.to_out = nn.Linear(hidden_size*4, class_count)

    def forward(self, x):
        _, h_n = self.recurrent(x)
        return self.to_out(h_n)

class TwoDimensionalGRUSeq2Seq(nn.Module):
    # The big one...
    def __init__(self, input_size, embedding_size, hidden_size, patch_rows, patch_cols, latent_size=64, num_layers=1, forcing=0.0,
                 device=torch.device('cpu')):
        super(TwoDimensionalGRUSeq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols

        # Encoder
        self.encoder = UnderlyingTwoDimensionalGRU(input_size, embedding_size, hidden_size, omnidirectionality=True,
                                                   device=device)

        # Convert into latent space
        self.channel_reduction_row = nn.Conv2d(4, 1, (1, 1), device=device)
        self.channel_reduction_col = nn.Conv2d(4, 1, (1, 1), device=device)
        self.row_compress = nn.Linear(patch_cols * hidden_size, latent_size//2) # Note: row & col are concatenated together
        self.col_compress = nn.Linear(patch_rows * hidden_size, latent_size//2)

        # VAE components
        self.latent_to_logvar = nn.Linear(latent_size, latent_size)
        self.latent_to_mean = nn.Linear(latent_size, latent_size)

        # A reparameterization is implied here

        # Convert out of latent space
        self.row_decompress = nn.Linear(latent_size//2, patch_cols * hidden_size)
        self.col_decompress = nn.Linear(latent_size//2, patch_rows * hidden_size)

        # Decoder
        self.decoder = UnderlyingTwoDimensionalGRU(input_size + 2, embedding_size, hidden_size,
                                                   omnidirectionality=False, device=device)
        self.evaluator = nn.Sequential(
            nn.Linear(hidden_size, input_size, device=device),
            nn.Sigmoid()
        )

        self.device = device
        self.forcing = forcing

    def forward(self, x, just_latent=False, just_decoder=False):
        batch_size = x.size()[0]
        if not just_decoder:
            _, pr, pc, bd = x.size()
            output, _, final_row, final_col = self.encoder(x, return_final_row_col=True)

            # Translate final row and final column of the encoder from describing the image
            # omnidirectionally to describing a combination of those directions in a single tensor
            final_row = self.channel_reduction_row(
                torch.stack([final_row[key] for key in self.encoder.direction_ref]).permute(1, 0, 2, 3))[:, 0, :, :]
            final_col = self.channel_reduction_col(
                torch.stack([final_col[key] for key in self.encoder.direction_ref]).permute(1, 0, 2, 3))[:, 0, :, :]

            # Reshape vectors down into a single column vector -> then compress
            final_row = self.row_compress(final_row.view(batch_size, pc * self.hidden_size))
            final_col = self.col_compress(final_col.view(batch_size, pr * self.hidden_size))

            # Concatenate to obtain latent vector
            latent = torch.cat((final_row, final_col), dim=1)
            logvar = self.latent_to_logvar(latent)
            mean = self.latent_to_mean(latent)
            if just_latent:
                return logvar, mean
        else:
            logvar, mean = x

        # Reparameterize into a probability distribution
        samp = self.reparameterize(logvar, mean)

        # Expand latent vector back out
        latent_row = self.row_decompress(samp[:, :self.latent_size//2])
        latent_col = self.col_decompress(samp[:, self.latent_size//2:])

        latent_row = latent_row.view(batch_size, self.patch_cols, self.hidden_size)
        latent_col = latent_col.view(batch_size, self.patch_rows, self.hidden_size)

        # Now, for the decoder part - attempt to reconstruct the image
        # First column counts as a "start of image" column -> cropped out at the end
        # reconstructed = torch.zeros((batch_size, pr, pc + 1, bd))
        replica = [[] for s in range(batch_size)]
        for s in range(batch_size):
            replica[s] = [[] for b in range(self.patch_rows)]

        # Store all of the hidden data on a grid - including our final row and column data
        # Note hidden_grid[:, 0, 0, :] is unused data
        hidden_grid = torch.zeros((batch_size, self.patch_rows + 1, self.patch_cols + 1, self.hidden_size), device=self.device)
        hidden_grid[:, 0, 1:, :] = latent_row
        hidden_grid[:, 1:, 0, :] = latent_col

        for row in range(self.patch_rows):
            for s in range(batch_size):
                replica[s][row].append(torch.zeros(self.input_size, device=self.device))
            for col in range(self.patch_cols):
                above = hidden_grid[:, row, col + 1, :]
                left = hidden_grid[:, row + 1, col, :]

                # Grab patch to the left to compute new hidden vector
                absolute_pos = torch.tensor([row / self.patch_rows * 2 - 1, col / self.patch_rows * 2 - 1] * batch_size,
                                            device=self.device).view(batch_size, 2)
                prev_input = torch.stack([replica[s][row][col] for s in range(batch_size)])
                prev_input = torch.cat((prev_input, absolute_pos), dim=1)

                h_ij = self.decoder.single_step(prev_input, above, left, batch_size, "Top-Left")
                hidden_grid[:, row + 1, col + 1, :] = h_ij

                # Convert hidden vector to a next-patch prediction
                # Tanh is applied at the end to make values between 0 and 1 -> for BCE Loss
                x_next = self.evaluator(h_ij)

                for s in range(batch_size):
                    rand = torch.rand(1).item()
                    if rand > self.forcing:
                        replica[s][row].append(x_next[s])
                    else:
                        replica[s][row].append((x[s, row, col, :] + 1) / 2) # force value from x

        # fully stack replica into a tensor
        rep = []
        for s in range(batch_size):
            cols = [torch.stack(replica[s][row]) for row in range(self.patch_rows)]
            full_im = torch.stack(cols)
            rep.append(full_im)
        rep = torch.stack(rep)

        return rep[:, :, 1:, :], logvar, mean

    def reparameterize(self, logvar, mean):
        coef = torch.randn_like(logvar).to(self.device)
        return mean + logvar * coef

    def to_latent(self, x):
        return self.forward(x, just_latent=True)

    def sample_image(self, logvar, mean):
        return self.forward((logvar, mean), just_decoder=True)

def save_checkpoint(input_size, embedding_size, hidden_size, num_layers, forcing, model, model_name):

    config = {
        'input_size': input_size,
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'forcing': forcing
    }

    checkpoint = {
        'config': config,
        'model': model.state_dict()
    }

    torch.save(checkpoint, f'Models/{model_name}')

if __name__ == '__main__':
    # net = OmniDirectionalTwoDimensionalGRU(9, 10, 12, omnidirectionality=False)

    # net = TwoDimensionalGRU(9, 12, 2, embedding_size=10, omnidirectionality=True)
    # net = UnderlyingTwoDimensionalGRU(9, 10, 12,  omnidirectionality=False)

    test_tensor = torch.rand((5, 2, 3, 9))
    # output, test_result, final_row, final_col = net.forward(test_tensor, return_final_row_col=True)
    #
    # print("Verifying output row:")
    # print(output[0, 0, 1, :, :])
    # print(final_row["Top-Left"][0])
    #
    # print("Verifying output col:")
    # print(output[0, 0, :, 2, :])
    # print(final_col["Top-Left"][0])


    # labels = torch.rand((4,1))
    # args = torch.argmax(test_result)

    net = TwoDimensionalGRUSeq2Seq(9, 10, 12, 2, 3, forcing=0.0)
    repla, logvar, mean = net(test_tensor)

    # latent = net.to_latent(test_tensor)
    # to_im = net.to_image(latent)

    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    print("hi")
