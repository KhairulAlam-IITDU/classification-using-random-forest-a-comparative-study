package twrf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class Node {
	
	Node() {}
	
	public Node leftChild;
	public Node rightChild;
	
	public List<List<Double>> attribute = new ArrayList<List<Double>>();
	public List<Double> decision = new ArrayList<>();
	
	public double Gain;
	public double nodeValue;
	public int featureIndexRow;
	public int featureIndexColumn;
	Map<Double, Double> classProb = new HashMap<>();
	
	Node(Node leftChild, Node rightChild, double Gain) {
		
		this.leftChild = leftChild;
		this.rightChild = rightChild;
		this.Gain = Gain;
	}
	
	Node(double value){
		this.nodeValue = value;
	}
	
}
