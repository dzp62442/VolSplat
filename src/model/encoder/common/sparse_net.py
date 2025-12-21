import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

class SparseGaussianHead(nn.Module):
    def __init__(self, in_channels=164, out_channels=38):

        super().__init__()
        
        self.num_gaussian_parameters = out_channels
        
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            dimension=3
        )
        self.act = ME.MinkowskiGELU()
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1,
            dimension=3
        )
    
        self.init_weights()
    
    def forward(self, sparse_input: ME.SparseTensor):
  
        x = self.conv1(sparse_input)
        x = self.act(x)
        x = self.conv2(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                try:
                    ME.utils.kaiming_normal_(m.kernel,
                                                mode='fan_out',
                                                nonlinearity='relu')
                except Exception:
                    nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
             
                if hasattr(m, 'bias') and m.bias is not None:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass

         
            elif isinstance(m, ME.MinkowskiBatchNorm):
                if hasattr(m, 'bn'):
                    try:
                        nn.init.constant_(m.bn.weight, 1)
                        nn.init.constant_(m.bn.bias, 0)
                    except Exception:
                        pass






class AttentionBlock(nn.Module):
    """基于Flash Attention的AttentionBlock"""
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}")
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = ME.MinkowskiBatchNorm(channels)
        
        self.qkv = ME.MinkowskiLinear(channels, channels * 3)
        self.proj_out = ME.MinkowskiLinear(channels, channels)

    def _attention(self, qkv: torch.Tensor):
        length, width = qkv.shape
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3)  # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1)  # (1, num_heads, length, ch)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                values = F.scaled_dot_product_attention(q, k, v)[0]
        else:
            values = F.scaled_dot_product_attention(q, k, v)[0]
        
        values = values.permute(1, 0, 2).reshape(length, -1)
        return values

    def forward(self, x: ME.SparseTensor):
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        
        feature_dense = self._attention(qkv.F)
        feature = ME.SparseTensor(
            features=feature_dense,
            coordinate_map_key=qkv.coordinate_map_key,
            coordinate_manager=qkv.coordinate_manager
        )
        
        output = self.proj_out(feature)
        return output + x 

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            dimension=3
        )
        self.norm = ME.MinkowskiBatchNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.conv(x)))

class SparseUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.upconv = ME.MinkowskiConvolutionTranspose(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            dimension=3
        )
        self.norm = ME.MinkowskiBatchNorm(out_channels)
        self.act = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x: ME.SparseTensor):
        return self.act(self.norm(self.upconv(x)))

class SparseUNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4, use_attention=False):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList() 
        self.use_attention = use_attention
        
        self.encoder_channels = []
        
        current_ch = in_channels
        for i in range(num_blocks):
            out_ch = 64 * (2 ** i)  
            self.encoders.append(SparseConvBlock(current_ch, out_ch, kernel_size=3, stride=2))
            self.encoder_channels.append(out_ch)  
            current_ch = out_ch

      
        bottleneck_in = current_ch
        bottleneck_out = bottleneck_in * 2
        
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(SparseConvBlock(bottleneck_in, bottleneck_out, kernel_size=3, stride=2))
        
        if use_attention:
            self.bottleneck.append(AttentionBlock(bottleneck_out))
        
        self.bottleneck.append(SparseConvBlock(bottleneck_out, bottleneck_out, kernel_size=3, stride=1))

        current_decoder_ch = bottleneck_out
        
        for i in range(num_blocks):
            decoder_out = self.encoder_channels[-1-i]
            
            upconv = SparseUpConvBlock(current_decoder_ch, decoder_out, kernel_size=3, stride=2)
            
            after_cat = ME.MinkowskiConvolution(
                decoder_out + self.encoder_channels[-1-i], 
                decoder_out, 
                kernel_size=1, 
                stride=1, 
                dimension=3
            )
            
            self.decoder_blocks.append(nn.ModuleList([upconv, after_cat]))
            
            current_decoder_ch = decoder_out

        self.final_upsample = SparseUpConvBlock(
            self.encoder_channels[0], 
            out_channels,              
            kernel_size=3, 
            stride=2
        )

    def forward(self, x: ME.SparseTensor):
        encoder_outputs = []
        
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        for layer in self.bottleneck:
            x = layer(x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            upconv, after_cat = decoder_block
            
            x = upconv(x)
            
            enc_index = len(encoder_outputs) - i - 1
            if enc_index >= 0 and enc_index < len(encoder_outputs):
                x = ME.cat(x, encoder_outputs[enc_index])
                x = after_cat(x)
        
        output = self.final_upsample(x)
        del encoder_outputs
        
        return output
