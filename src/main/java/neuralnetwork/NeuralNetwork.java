package neuralnetwork;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralNetwork {

	// Define Hyperparameters
	private Integer inputLayerSize;
	private Integer outputLayerSize;
	private Integer hiddenLayerSize;
	// Weights (parameters)
	private INDArray weightMatrix1;
	private INDArray weightMatrix2;

	private INDArray z2;
	private INDArray z3;
	private INDArray a2;

	public NeuralNetwork(Integer inputLayerSize, Integer outputLayerSize,
			Integer hiddenLayerSize) {
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.weightMatrix1 = Nd4j.randn(inputLayerSize, hiddenLayerSize);
		this.weightMatrix2 = Nd4j.randn(hiddenLayerSize, outputLayerSize);
	}

	/**
	 * Propogate inputs though network
	 * 
	 * @param X
	 */
	public INDArray forward(INDArray X) {
		this.z2 = X.mmul(this.weightMatrix1);
		this.a2 = sigmoid(z2);
		this.z3 = a2.mmul(this.weightMatrix2);
		INDArray yHat = sigmoid(z3);
		return yHat;
	}

	/**
	 * Gradient of sigmoid
	 * 
	 * @param z
	 */
	protected INDArray sigmoidPrime(INDArray z) {
		INDArray leftSide = exp(z.mul(-1));
		INDArray rightSide = pow(exp(z.mul(-1)).add(1), 2);
		return leftSide.div(rightSide);
	}

	/**
	 * Compute cost for given X,y, use weights already stored in class.
	 * 
	 * @param X
	 * @param y
	 */
	protected INDArray costFunction(INDArray X, INDArray y) {
		INDArray yHat = forward(X);
		INDArray J = ((pow(y.sub(yHat), 2)).sum(0)).mul(0.5);
		return J;
	}

	/**
	 * Compute derivative with respect to W and W2 for a given X and y
	 * 
	 * @param X
	 * @param y
	 */
	protected INDArray[] costFunctionPrime(INDArray X, INDArray y) {
		INDArray yHat = forward(X);

		INDArray leftSide = (y.sub(yHat)).mul(-1);
		INDArray rightSide = sigmoidPrime(this.z3);
		INDArray delta3 = leftSide.mul(rightSide);
		INDArray djdW2 = this.a2.transpose().mmul(delta3);

		INDArray leftSide2 = delta3.mmul(this.weightMatrix2.transpose());
		INDArray rightSide2 = sigmoidPrime(this.z2);
		INDArray delta2 = leftSide2.mul(rightSide2);
		INDArray djdW1 = X.transpose().mmul(delta2);

		return new INDArray[] { djdW1, djdW2 };
	}

	// Getter & Setter Methods

	public Integer getInputLayerSize() {
		return inputLayerSize;
	}

	public void setInputLayerSize(Integer inputLayerSize) {
		this.inputLayerSize = inputLayerSize;
	}

	public Integer getOutputLayerSize() {
		return outputLayerSize;
	}

	public void setOutputLayerSize(Integer outputLayerSize) {
		this.outputLayerSize = outputLayerSize;
	}

	public Integer getHiddenLayerSize() {
		return hiddenLayerSize;
	}

	public void setHiddenLayerSize(Integer hiddenLayerSize) {
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public INDArray getWeightMatrix1() {
		return weightMatrix1;
	}

	public void setWeightMatrix1(INDArray weightMatrix1) {
		this.weightMatrix1 = weightMatrix1;
	}

	public INDArray getWeightMatrix2() {
		return weightMatrix2;
	}

	public void setWeightMatrix2(INDArray weightMatrix2) {
		this.weightMatrix2 = weightMatrix2;
	}

}
