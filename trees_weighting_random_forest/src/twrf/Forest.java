package twrf;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;


public class Forest {

	public ArrayList <DecisionTree> totalTrees = new ArrayList<>();
	public ArrayList<Double> tpeValues = new ArrayList<>();
	public double totalTpe = 0.0;

	
	public void makeDecisionTree(DataNode rtnd, int op) {

		int option = op;
		
		for(int i = 0; i < 100; i++) {

			DataNode[] inOutData = bootstrap(rtnd);

			int numberOfFeatures = inOutData[0].features.get(0).size();

			int featuresForSingleTree = (int)Math.ceil(Math.sqrt(numberOfFeatures));
			
			//int featuresForSingleTree = (int)(Math.log10(numberOfFeatures)/Math.log10(2))+1;

			DecisionTree  decisionTree = new DecisionTree(inOutData[0], featuresForSingleTree);

			//decisionTree.inorderTraverse(decisionTree.root);
			
			if(option == 2){
				decisionTree.weight = getOOBAccuracyRate(decisionTree, inOutData[1]);
			}
			else if(option == 1){
				double tmp = getTreePredictionErrorRate(decisionTree, inOutData[1]);
				tmp = Math.pow((1.0/tmp), 5);
				totalTpe = totalTpe + tmp;
				tpeValues.add(tmp);
			}
			
			totalTrees.add(decisionTree);

		}
		
		if(option == 1){
			for(int i = 0; i < 100; i++) {
				
				totalTrees.get(i).weight = tpeValues.get(i)/totalTpe;
			}
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
    
    
    private double getTreePredictionErrorRate(DecisionTree decisionTree,
			DataNode inOutData) {

    	double tpeRate = 0.0;
    	double oobTotal = 0.0;
    	
    	for(int i = 0; i < inOutData.features.size(); i++){
    		
    		double key = decisionTree.traverse(decisionTree.root, inOutData.features.get(i));		

    		if(key != inOutData.labels.get(i)){
    			oobTotal = oobTotal + 1;		//Math.abs(key - inOutData.labels.get(i))		
    		}
    	}
    	
    	tpeRate = oobTotal/(double)inOutData.features.size();
		
    	return tpeRate;
	}


	public DataNode[] bootstrap(DataNode data) {
        
    	Random random = new Random();
    	//random.setSeed((long) 1);
    	
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
			
			System.out.println("\nOPSS! Blank Forest Probability!");
		}

		return finalDecision;
	}

}
