package twrf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class EntropyCalculation {

	public boolean zeroEntropy = false;
	
	//public  TempNode maxGainedElement(Node subset) //old

	public Node maxGainedElement(List<List<Double>> feature, List<Double> label, int featureSubsetSize) {	

		//System.out.println("Entered inside maxGainedElement");
		
		Node element = new Node();
		
		int row = 0, col = 0;
		
		//'new added'
		
		Map<Double, Integer> decision = new HashMap<>(); 
		
		//int totalMembers = subset.attribute.size();
		
		for(Double ab : label){
			
			if(decision.containsKey(ab)){
				
				int count = decision.get(ab);
				decision.put(ab, count+1);
			}
			else
				decision.put(ab, 1);
				
		}
		
		//end of 'new added'
		
		double parentEntropy = evaluateEntropy(decision);	
		
		if(parentEntropy == 0.0){
			
			this.zeroEntropy = true;

			return null;
		}
		
		double max = -10;

		//System.out.println(subset.attribute.get(0).size());
		//System.out.println(subset.attribute.size());
		
		Random rand = new Random(); //new
		
		//rand.setSeed((long) 1);
		
		List<Integer> selectedFeatures = new ArrayList<>();
		
		while(selectedFeatures.size() != featureSubsetSize){
			
			int column = rand.nextInt(feature.get(0).size());
			
			if(selectedFeatures.contains(column)){
				continue;
			}
			else{
				selectedFeatures.add(column);
			}
		}
		
		for(int i = 0; i < selectedFeatures.size(); i++) {
			
			List<Double> singleFeature = new ArrayList<>(); //new
			
			//int column = rand.nextInt(feature.get(0).size());
			
			int column = selectedFeatures.get(i);
			//System.out.print("Picked column: " + column + ", ");
			
			for(int j = 0; j < feature.size(); j++) {

				//System.out.println("Single feature: " + subset.attribute.get(j).get(column));
				
				singleFeature.add(feature.get(j).get(column));
			}		

			Node n = evaluateGain(label, parentEntropy, singleFeature);


			if(max < n.Gain) { // max <= n.Gain
			
				max = n.Gain;
				
				row = n.featureIndexRow;
				col = column;
				
			}
			//System.out.println(n.Gain +"\n");

		}
		//System.out.println();
		//element = subset;
		element.featureIndexColumn = col;
		element.featureIndexRow = row;

		return element;
	}

	
	
	private Node evaluateGain(List<Double> label, double parentEntropy, List<Double> featureColumn) {

		Node n = new Node();
		
		double bestGain = -1.0;
		double threshold, leq, gr;
		double l, g;
		
		Map<Double, Integer> lessEqual = new HashMap<>();
		Map<Double, Integer> greater = new HashMap<>();
		
		//int totalMembers = subset.attribute.size();
		
		for(int i = 0; i < featureColumn.size(); i++) {

			threshold = featureColumn.get(i);
			
			leq = 0.0;
			gr = 0.0;
			
			lessEqual.clear();
			greater.clear();
			
			//double decision = subset.decision.get(i);

			for(int j = 0; j < featureColumn.size(); j++) {

				if(threshold < featureColumn.get(j)) {

					
					gr++;
					
					if(greater.containsKey(label.get(j))){
						
						int count = greater.get(label.get(j));
						greater.put(label.get(j), count+1);
					}
					else
						greater.put(label.get(j), 1);
					
				}
				else {

					leq++;
					
					if(lessEqual.containsKey(label.get(j))){
						
						int count = lessEqual.get(label.get(j));
						lessEqual.put(label.get(j), count+1);
					}
					else
						lessEqual.put(label.get(j), 1);
			
				}

			}

			l = evaluateEntropy(lessEqual);
			g = evaluateEntropy(greater);

			double avgChildEntropy = (((leq/(double)featureColumn.size())*l)+((gr/(double)featureColumn.size())*g));
			
			double Gain = parentEntropy - avgChildEntropy;

			if(Gain > bestGain) {

				bestGain = Gain;

				n.featureIndexRow = i;
				n.Gain = bestGain;
			}

		}

		return n;
	}

	private  double evaluateEntropy(Map<Double, Integer> subsetData) {
		
		int distinctDecision = subsetData.size();
		
		int[] count = new int[distinctDecision];
		int n = 0;
		
		int total = 0;
		double entropy=0.0;
		
		Set<Double> keys = subsetData.keySet();
		
		for(Double key : keys){
			
			count[n] = subsetData.get(key);
			
			total += count[n];
			
			//System.out.println(count[n]+"\t"+total);
			
			n++;
		}
		
		for(int i = 0; i < distinctDecision; i++){
			
			//System.out.println((double)count[i]);
			
			entropy += (-((double)count[i]/(double)total)*(Math.log10((double)count[i]/(double)total)/Math.log10(2)));
		}
		
		
		return entropy;
	}
}
