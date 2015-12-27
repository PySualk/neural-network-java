package neuralnetwork;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralNetworkTest {

	private NeuralNetwork nn;
	private INDArray X;
	private INDArray y;

	private Integer inputLayerSize = 2;
	private Integer outputLayerSize = 1;
	private Integer hiddenLayerSize = 3;
	float delta = 0.00001f;

	@Before
	public void setup() {
		nn = new NeuralNetwork(inputLayerSize, outputLayerSize, hiddenLayerSize);
		X = Nd4j.create(new float[] { 3, 5, 5, 1, 10, 2 }, new int[] { 3, 2 });
		y = Nd4j.create(new float[] { 75, 82, 93 }, new int[] { 3, 1 });

		// Normalize
		X.diviRowVector(Nd4j.max(X, 0));
		y.divi(Nd4j.create(new float[] { 100 }, new int[] { 1 }));

		// Use fixed values for weights
		nn.setWeightMatrix1(Nd4j.create(new float[] { -0.24743783f,
				-2.2297615f, -0.71937955f, -0.62020646f, 1.37920321f,
				0.09350954f }, new int[] { inputLayerSize, hiddenLayerSize }));
		nn.setWeightMatrix2(Nd4j.create(new float[] { 1.28491091f,
				-2.18504611f, -0.71523799f }, new int[] { hiddenLayerSize,
				outputLayerSize }));
	}

	@Test
	public void canBeInitialized() {
		assertThat(nn.getInputLayerSize(), equalTo(2));
		assertThat(nn.getHiddenLayerSize(), equalTo(3));
		assertThat(nn.getOutputLayerSize(), equalTo(1));
	}

	@Test
	public void inputCanBeForwarded() {
		INDArray yHat = nn.forward(X);
		assertEquals(0.20216f, yHat.getFloat(0), delta);
		assertEquals(0.40293f, yHat.getFloat(1), delta);
		assertEquals(0.47563f, yHat.getFloat(2), delta);
	}

	@Test
	public void costFunctionReturnsCorrectValues() {
		INDArray J = nn.costFunction(X, y);
		assertEquals(0.34026f, J.getFloat(0), delta);
	}

	@Test
	public void sigmoidDerivativeReturnsCorrectValues() {
		INDArray z = Nd4j.create(new double[] { -0.29d, 0, 1 }, new int[] { 3,
				1 });
		INDArray result = nn.sigmoidPrime(z);
		assertEquals(0.24482f, result.getFloat(0), delta);
		assertEquals(0.25f, result.getFloat(1), delta);
		assertEquals(0.19661f, result.getFloat(2), delta);
	}

	@Test
	public void costFunctionPrimeReturnsCorrectValues() {
		INDArray[] result = nn.costFunctionPrime(X, y);
		assertEquals(-0.05769f, result[0].getFloat(0), delta);
		assertEquals(-0.11631f, result[1].getFloat(0), delta);
	}
}
