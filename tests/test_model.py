"""
Unit tests for Sparse SSM model components.

This module provides tests to verify the correctness of various model components,
including discretization, HiPPO initialization, selective parameterization, and
the full model architecture.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.discretization import (
    discretize_zoh, 
    discretize_bilinear, 
    discretize_generalized_bilinear,
    DiscretizationLayer
)
from src.model.hippo import (
    make_hippo_legs,
    make_hippo_legt,
    make_hippo_fourier,
    HiPPOInit
)
from src.model.selective_param import (
    DenseSelectiveParameterization,
    SparseSelectiveParameterization,
    LowRankSelectiveParameterization
)
from src.model.ssm import SSMLayer, SSMKernel, ParallelSSMKernel
from src.model.mamba import SparseSSM


class TestDiscretization(unittest.TestCase):
    """Test discretization methods."""
    
    def setUp(self):
        """Set up test parameters."""
        self.n = 4  # State dimension
        self.m = 1  # Input dimension
        self.A = torch.randn(self.n, self.n)
        self.B = torch.randn(self.n, self.m)
        self.dt = 0.01
    
    def test_zoh(self):
        """Test zero-order hold discretization."""
        A_d, B_d = discretize_zoh(self.A, self.B, self.dt)
        
        # Check shapes
        self.assertEqual(A_d.shape, (self.n, self.n))
        self.assertEqual(B_d.shape, (self.n, self.m))
        
        # Test with batch dimension
        batch_size = 2
        A_batch = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        B_batch = self.B.unsqueeze(0).expand(batch_size, -1, -1)
        
        A_d_batch, B_d_batch = discretize_zoh(A_batch, B_batch, self.dt)
        
        # Check batch shapes
        self.assertEqual(A_d_batch.shape, (batch_size, self.n, self.n))
        self.assertEqual(B_d_batch.shape, (batch_size, self.n, self.m))
    
    def test_bilinear(self):
        """Test bilinear (Tustin) discretization."""
        A_d, B_d = discretize_bilinear(self.A, self.B, self.dt)
        
        # Check shapes
        self.assertEqual(A_d.shape, (self.n, self.n))
        self.assertEqual(B_d.shape, (self.n, self.m))
        
        # Test with batch dimension
        batch_size = 2
        A_batch = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        B_batch = self.B.unsqueeze(0).expand(batch_size, -1, -1)
        
        A_d_batch, B_d_batch = discretize_bilinear(A_batch, B_batch, self.dt)
        
        # Check batch shapes
        self.assertEqual(A_d_batch.shape, (batch_size, self.n, self.n))
        self.assertEqual(B_d_batch.shape, (batch_size, self.n, self.m))
    
    def test_discretization_layer(self):
        """Test DiscretizationLayer."""
        layer = DiscretizationLayer(
            method='zoh',
            dt_init=0.01,
            dt_learnable=True
        )
        
        A_d, B_d = layer(self.A, self.B)
        
        # Check shapes
        self.assertEqual(A_d.shape, (self.n, self.n))
        self.assertEqual(B_d.shape, (self.n, self.m))
        
        # Test with different method
        layer = DiscretizationLayer(
            method='bilinear',
            dt_init=0.01,
            dt_learnable=False
        )
        
        A_d, B_d = layer(self.A, self.B)
        
        # Check shapes
        self.assertEqual(A_d.shape, (self.n, self.n))
        self.assertEqual(B_d.shape, (self.n, self.m))


class TestHiPPO(unittest.TestCase):
    """Test HiPPO initialization methods."""
    
    def setUp(self):
        """Set up test parameters."""
        self.N = 4  # State dimension
    
    def test_hippo_legs(self):
        """Test HiPPO-LegS initialization."""
        A, B = make_hippo_legs(self.N)
        
        # Check shapes
        self.assertEqual(A.shape, (self.N, self.N))
        self.assertEqual(B.shape, (self.N, 1))
        
        # Check properties
        self.assertTrue(torch.all(torch.isfinite(A)))
        self.assertTrue(torch.all(torch.isfinite(B)))
    
    def test_hippo_legt(self):
        """Test HiPPO-LegT initialization."""
        A, B = make_hippo_legt(self.N)
        
        # Check shapes
        self.assertEqual(A.shape, (self.N, self.N))
        self.assertEqual(B.shape, (self.N, 1))
        
        # Check properties
        self.assertTrue(torch.all(torch.isfinite(A)))
        self.assertTrue(torch.all(torch.isfinite(B)))
    
    def test_hippo_fourier(self):
        """Test HiPPO-Fourier initialization."""
        A, B = make_hippo_fourier(self.N)
        
        # Check shapes (Fourier requires even N)
        if self.N % 2 == 0:
            self.assertEqual(A.shape, (self.N, self.N))
            self.assertEqual(B.shape, (self.N, 1))
            
            # Check properties
            self.assertTrue(torch.all(torch.isfinite(A)))
            self.assertTrue(torch.all(torch.isfinite(B)))
    
    def test_hippo_init_module(self):
        """Test HiPPOInit module."""
        for method in ['legs', 'legt', 'fourier']:
            hippo = HiPPOInit(
                state_dim=self.N,
                method=method,
                normalize=True,
                trainable=True
            )
            
            A, B = hippo()
            
            # Check shapes
            self.assertEqual(A.shape, (self.N, self.N))
            self.assertEqual(B.shape, (self.N, 1))
            
            # Check properties
            self.assertTrue(torch.all(torch.isfinite(A)))
            self.assertTrue(torch.all(torch.isfinite(B)))
            
            # Check if parameters are trainable
            self.assertTrue(hippo.A.requires_grad)
            self.assertTrue(hippo.B.requires_grad)


class TestSelectiveParameterization(unittest.TestCase):
    """Test selective parameterization modules."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 2
        self.seq_len = 3
        self.input_dim = 16
        self.hidden_dim = 16
        self.state_dim = 4
        
        # Create input tensor
        self.input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
    
    def test_dense_selective_parameterization(self):
        """Test DenseSelectiveParameterization."""
        module = DenseSelectiveParameterization(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        )
        
        params = module(self.input)
        
        # Check that all parameters are present
        self.assertIn('A', params)
        self.assertIn('B', params)
        self.assertIn('C', params)
        self.assertIn('D', params)
        
        # Check shapes
        self.assertEqual(params['A'].shape, (self.batch_size, self.seq_len, self.state_dim, self.state_dim))
        self.assertEqual(params['B'].shape, (self.batch_size, self.seq_len, self.state_dim, 1))
        self.assertEqual(params['C'].shape, (self.batch_size, self.seq_len, 1, self.state_dim))
        self.assertEqual(params['D'].shape, (self.batch_size, self.seq_len, 1, 1))
    
    def test_sparse_selective_parameterization(self):
        """Test SparseSelectiveParameterization."""
        module = SparseSelectiveParameterization(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            sparsity_level=0.9
        )
        
        params = module(self.input)
        
        # Check that all parameters are present
        self.assertIn('A', params)
        self.assertIn('B', params)
        self.assertIn('C', params)
        self.assertIn('D', params)
        
        # Check shapes
        self.assertEqual(params['A'].shape, (self.batch_size, self.seq_len, self.state_dim, self.state_dim))
        self.assertEqual(params['B'].shape, (self.batch_size, self.seq_len, self.state_dim, 1))
        self.assertEqual(params['C'].shape, (self.batch_size, self.seq_len, 1, self.state_dim))
        self.assertEqual(params['D'].shape, (self.batch_size, self.seq_len, 1, 1))
        
        # Check sparsity
        sparsity = (params['A'] == 0).float().mean().item()
        self.assertGreater(sparsity, 0.5)  # Should be sparse
    
    def test_low_rank_selective_parameterization(self):
        """Test LowRankSelectiveParameterization."""
        module = LowRankSelectiveParameterization(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            rank=2
        )
        
        params = module(self.input)
        
        # Check that all parameters are present
        self.assertIn('A', params)
        self.assertIn('B', params)
        self.assertIn('C', params)
        self.assertIn('D', params)
        
        # Check shapes
        self.assertEqual(params['A'].shape, (self.batch_size, self.seq_len, self.state_dim, self.state_dim))
        self.assertEqual(params['B'].shape, (self.batch_size, self.seq_len, self.state_dim, 1))
        self.assertEqual(params['C'].shape, (self.batch_size, self.seq_len, 1, self.state_dim))
        self.assertEqual(params['D'].shape, (self.batch_size, self.seq_len, 1, 1))


class TestSSMKernel(unittest.TestCase):
    """Test SSM kernel implementations."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 2
        self.seq_len = 3
        self.hidden_dim = 16
        self.state_dim = 4
        
        # Create input tensor
        self.input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
    
    def test_ssm_kernel(self):
        """Test SSMKernel."""
        kernel = SSMKernel(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            selective_param_class='dense'
        )
        
        output = kernel(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Test with initial state
        state = torch.zeros(self.batch_size, self.state_dim)
        kernel.return_state = True
        output, final_state = kernel(self.input, state)
        
        # Check output and state shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(final_state.shape, (self.batch_size, self.state_dim))
    
    def test_parallel_ssm_kernel(self):
        """Test ParallelSSMKernel."""
        kernel = ParallelSSMKernel(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            selective_param_class='dense'
        )
        
        output = kernel(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Test with initial state
        state = torch.zeros(self.batch_size, self.state_dim)
        kernel.return_state = True
        output, final_state = kernel(self.input, state)
        
        # Check output and state shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(final_state.shape, (self.batch_size, self.state_dim))


class TestSSMLayer(unittest.TestCase):
    """Test SSM layer implementation."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 2
        self.seq_len = 3
        self.hidden_dim = 16
        self.state_dim = 4
        
        # Create input tensor
        self.input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
    
    def test_ssm_layer(self):
        """Test SSMLayer."""
        layer = SSMLayer(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim
        )
        
        output = layer(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))


class TestSparseSSM(unittest.TestCase):
    """Test the complete SparseSSM model."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 2
        self.seq_len = 3
        self.hidden_dim = 16
        self.state_dim = 4
        self.ffn_dim = 32
        self.num_layers = 2
        self.vocab_size = 100
        
        # Create input tensor
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    
    def test_sparse_ssm_forward(self):
        """Test SparseSSM forward pass."""
        model = SparseSSM(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            max_seq_len=self.seq_len,
            selective_param_class='sparse'
        )
        
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        
        # Check that logits are returned
        self.assertIn('logits', outputs)
        
        # Check logits shape
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.vocab_size))
    
    def test_sparse_ssm_generate(self):
        """Test SparseSSM text generation."""
        model = SparseSSM(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            ffn_dim=self.ffn_dim,
            num_layers=self.num_layers,
            max_seq_len=10,  # Longer max_seq_len for generation
            selective_param_class='sparse'
        )
        
        # Generate text
        generated_ids = model.generate(
            input_ids=self.input_ids,
            max_length=5,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Check generated_ids shape
        self.assertEqual(generated_ids.shape[0], self.batch_size)
        self.assertGreaterEqual(generated_ids.shape[1], self.seq_len)
        self.assertLessEqual(generated_ids.shape[1], 5)


if __name__ == '__main__':
    unittest.main()
