package wrfid;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;


public class Forest {

	public ArrayList <DecisionTree> totalTrees = new ArrayList<>();

	public void makeDecisionTree(DataNode rtnd, Map<Double, Double> clsWght) { //rtnd

		for(int i = 0; i < 100; i++) {

			DataNode[] inOutData = bootstrap(rtnd);
			
			//System.out.println("\nTree " + (i+1) + " is building...");

			int numberOfFeatures = inOutData[0].features.get(0).size();

			int featuresForSingleTree = (int)Math.ceil(Math.sqrt(numberOfFeatures));
			
			//int featuresForSingleTree = (int)(Math.log10(numberOfFeatures)/Math.log10(2))+1;

			DecisionTree  decisionTree = new DecisionTree(inOutData[0], featuresForSingleTree, clsWght);

			//decisionTree.inorderTraverse(decisionTree.root);
			
			decisionTree.weight = getOOBAccuracyRate(decisionTree, inOutData[1]);
			
			totalTrees.add(decisionTree);

		}
	}
	
	
    private double getOOBAccuracyRate(DecisionTree decisionTree,
			DataNode dataNode) {
    	
    	int correctLabelCount = 0;
    	double accuracyRate = 0.0;
    	
    	for(int i = 0; i < dataNode.features.size(); i++){
    		double key = decisionTree.traverse(decisionTree.root, dataNode.features.get(i));
    		
    		if(key == dataNode.labels.get(i)){
    			correctLabelCount++;
    		}
    	}
    	
    	accuracyRate = (double)correctLabelCount/dataNode.labels.size();
		
    	return accuracyRate;
	}


	public DataNode[] bootstrap(DataNode data) {
        
    	Random random = new Random();
    	
    	DataNode[] bootstrappedData = new DataNode[2];
    	bootstrappedData[0] = new DataNode();
    	bootstrappedData[1] = new DataNode();
    	
    	List<Integer> trackInBag = new ArrayList<>();

        for (int i = 0; i < data.features.size(); i++) {
            
        	int index = random.nextInt(data.features.size());
        	trackInBag.add(index);
        	
        	bootstrappedData[0].features.add(data.features.get(index));
        	bootstrappedData[0].labels.add(data.labels.get(index));
        }
        
        for (int i = 0; i < data.features.size(); i++) {
        	
        	if(!trackInBag.contains(i)){
        		//System.out.print(i + ", ");
        		bootstrappedData[1].features.add(data.features.get(i));
        		bootstrappedData[1].labels.add(data.labels.get(i));
        	}
        }

        return bootstrappedData;
    }


	public double traverseForest(List<Double> testSet) {

		//int count = 0;

		Map<Double, Double> counter = new TreeMap<>();

		//loop start
		for(DecisionTree dt : totalTrees) {

			//count++;
			//System.out.println("\nTree number: " + count);
			
			double lbl = dt.traverse(dt.root, testSet);
			
			//start majority vote
			if(counter.containsKey(lbl)) {

				double value = counter.get(lbl);
				counter.remove(lbl);
				counter.put(lbl, value+(1.0*dt.weight));

			}
			else {

				counter.put(lbl, (1.0*dt.weight));

			}			
			//end majority vote
			

		}//loop end

		double result = -100.0;
		double finalDecision = -100.0;

		if(!counter.isEmpty()) {
			
			for(Double key : counter.keySet()) {

				if(result < counter.get(key)) {

					result = counter.get(key);

					finalDecision = key;
				}
			}
		}
		else {
			
			System.out.println("\nOOPS! Blank Forest Probability!");
		}

		return finalDecision;
	}

}
