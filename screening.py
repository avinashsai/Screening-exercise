import pytest
from trainer import NewTrainer

# Check the data types of inputs to the train function
@pytest.mark.parametrize('x', [[[1, 2, 3, 4]], 
							[[0.11, 0.2, 0.3, 0.4]],
							[[0.11, 1, 0.3, 0.4]]])
@pytest.mark.parametrize('y', [[[1, 2, 3, 4]], 
							[[1, 2, 3, 4]]],
							[[0.11, 0.2, 0.3, 0.4]])
def test_types_train(x, y):
	trainer_class = NewTrainer()
	 with pytest.raises(Exception) as wrong_types:
	 	trainer_class.train(x, y)

# Check the data types of inputs to the test function
@pytest.mark.parametrize('x', [[[1, 2, 3, 4]], 
							[[0.11, 0.2, 0.3, 0.4]],
							[[0.11, 1, 0.3, 0.4]]])
def test_types_test(x, y):
	test_class = NewTrainer()
	 with pytest.raises(Exception) as wrong_types:
	 	test_class.train(x)

# Check if the input sizes match in the train function
@pytest.mark.parametrize('x', [[[1, 2, 3]], 
							[[0.11, 0.2, 0.3, 0.4]],
							[[]]])
@pytest.mark.parametrize('y', [[[1, 2, 3, 4]], 
							[[1, 2, 3]]],
							[[0.11, 0.2, 0.3, 0.4]])
def check_shapes(x, y):
	trainer_class = NewTrainer()
	 with pytest.raises(Exception) as wrong_shapes:
	 	trainer_class.train(x, y)

# Check the number of arguments in the train function
@pytest.mark.parametrize('x', [[[1, 2, 3, 4]], 
							[[0.11, 0.2, 0.3, 0.4]],
							[[0.11, 1, 0.3, 0.4]]])
def test_arguments_train(x):
	trainer_class = NewTrainer()
	 with pytest.raises(Exception) as wrong_arguments_types:
	 	trainer_class.train(x)

# Check the number of arguments of inputs to the test function
@pytest.mark.parametrize('x', [[[1, 2, 3, 4]], 
							[[0.11, 0.2, 0.3, 0.4]],
							[[0.11, 1, 0.3, 0.4]]])
@pytest.mark.parametrize('y', [[[1, 2, 3, 4]], 
							[[1, 2, 3, 4]]],
							[[0.11, 0.2, 0.3, 0.4]])
def test_types_train(x, y):
	trainer_class = NewTrainer()
	 with pytest.raises(Exception) as wrong_arguments_types:
	 	trainer_class.test(x, y)
