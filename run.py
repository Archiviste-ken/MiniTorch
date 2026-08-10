from tests.test_autograd import test_addition, test_multiplication, test_multiple_paths, numerical_gradient
from autograd.value import Value


if __name__ == "__main__":
    try:
        test_addition()
        print("✓ test_addition passed")
    except AssertionError as e:
        print(f"✗ test_addition failed: {e}")
    
    try:
        test_multiplication()
        print("✓ test_multiplication passed")
    except AssertionError as e:
        print(f"✗ test_multiplication failed: {e}")
    
    try:
        test_multiple_paths()
        print("✓ test_multiple_paths passed")
    except AssertionError as e:
        print(f"✗ test_multiple_paths failed: {e}")
    
    # Test numerical gradient
    try:
        # Test simple function: f(x) = x^2
        def f(x):
            return x * x
        
        x = Value(3.0)
        num_grad = numerical_gradient(f, x)
        
        # numerical_gradient returns a Value, extract the data
        num_grad_value = num_grad.data if isinstance(num_grad, Value) else num_grad
        
        # Analytical gradient: f(x) = x^2, f'(x) = 2x = 6
        analytical_grad = 2 * 3.0
        
        # Check if they're close (within tolerance)
        assert abs(num_grad_value - analytical_grad) < 1e-5, f"Gradient mismatch: {num_grad_value} vs {analytical_grad}"
        print(f"✓ numerical_gradient passed (num_grad={num_grad_value:.6f}, analytical={analytical_grad:.6f})")
    except Exception as e:
        print(f"✗ numerical_gradient failed: {e}")