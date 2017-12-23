package wrfid;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;

public class DecisionTree {

	public int attributeSubsetSize;
	public int traineeDataSize;
	public double weight;
	public Map<Double, Double> weights = new HashMap<>();
	public Node root;
	public double decisionClass = -1.0;

	public DecisionTree(DataNode dataNode, int featuresForSingleTree, Map<Double, Double> clsWght) {

		this.attributeSubsetSize = featuresForSingleTree;
		this.traineeDataSize = dataNode.labels.size();
		
		for(Double key : clsWght.keySet()){
			this.weights.put(key, clsWght.get(key));
		}
		
		this.root = makeTree(this.root, dataNode, this.attributeSubsetSize, 0);
	}


	private Node makeTree(Node currentNode, DataNode dataNode,
			int featureSubsetSize, int height) {

		if (dataNode.labels.size() < this.traineeDataSize / 25 || height > 7) {
			return new Node(majority(dataNode));
		}
		else{

			EntropyCalculation e1 = new EntropyCalculation();

			Node n = e1.maxGainedElement(dataNode.features, dataNode.labels, featureSubsetSize, this.weights); //new


			if(e1.zeroEntropy){

				currentNode = new Node(majority(dataNode));

				return currentNode;
			}
			else{

				currentNode = new Node();

				currentNode.featureIndexColumn = n.featureIndexColumn;
				currentNode.featureIndexRow = n.featureIndexRow;

				currentNode.attribute = dataNode.features;
				currentNode.decision = dataNode.labels;

				currentNode.nodeValue = dataNode.features.get(currentNode.featureIndexRow).get(currentNode.featureIndexColumn);

				currentNode.leftChild = new Node();
				currentNode.rightChild = new Node();

				DataNode leftNode = new DataNode();
				DataNode rightNode = new DataNode();

				for (int i = 0; i < dataNode.features.size(); i++) {

					if(currentNode.nodeValue >= dataNode.features.get(i).get(currentNode.featureIndexColumn)) {

						leftNode.features.add(dataNode.features.get(i));
						leftNode.labels.add(dataNode.labels.get(i));

					}
					else {

						rightNode.features.add(dataNode.features.get(i));
						rightNode.labels.add(dataNode.labels.get(i));
					}
				}

				currentNode.leftChild = makeTree(currentNode.leftChild, leftNode, featureSubsetSize, height+1);

				currentNode.rightChild = makeTree(currentNode.rightChild, rightNode, featureSubsetSize, height+1);

				return currentNode;
			}
		}
	}


	private double majority(DataNode dataNode) {

		HashMap<Double, Integer> map = new HashMap<>();

		for (double label : dataNode.labels) {

			if(map.containsKey(label)) {
				map.put(label, map.get(label)+1);
			}
			else {
				map.put(label, 1);
			}
		}

		double major = -1.0;
		double maxCount = -1.0;

		for (double label : map.keySet()) {

			if (((double)map.get(label)*weights.get(label)) > maxCount) {
				maxCount = ((double)map.get(label)*weights.get(label));
				major = label;
			}
		}
		return major;
	}

	public double traverse(Node current, List<Double> testSet) {

		if(current != null){

			if(current.leftChild == null && current.rightChild == null) {
				return current.nodeValue;
			}

			if(current.nodeValue >= testSet.get(current.featureIndexColumn)) {
				return traverse(current.leftChild, testSet);
			}
			else {
				return traverse(current.rightChild, testSet);
			}
		}
		else{
			return -1.0;
		}
	}


	public void inorderTraverse(Node root){

		if (root == null) {
			return;
		}

		Stack<Node> stack = new Stack<Node>();
		Node node = root;

		while (node != null) {
			stack.push(node);
			node = node.leftChild;
		}

		// traverse the tree
		while (stack.size() > 0) {

			node = stack.pop();

			if(node.leftChild == null && node.rightChild == null)
				System.out.print(node.nodeValue + ", ");

			if (node.rightChild != null) {
				node = node.rightChild;

				while (node != null) {
					stack.push(node);
					node = node.leftChild;
				}
			}
		}
	}

}
