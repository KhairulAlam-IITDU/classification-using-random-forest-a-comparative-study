package twrf;

import java.util.List;

public class Main {

	public static void main(String[] args) {

		ReadFile r = new ReadFile();

		r.fileReader();

		int folds = 10;

		DataNode[] data = new DataNode[folds];
		DataNode[] traineeData = new DataNode[folds];
		DataNode[] testData = new DataNode[folds];

		data = r.makeFolds(folds);

		//make k-folds trainee and test data sets
		for(int i = 0; i < folds; i++){

			traineeData[i] = new DataNode();
			testData[i] = new DataNode();

			for (int j = 0; j < folds; j++) {


				if(i == j){

					for(List<Double> dts : data[j].features){
						testData[i].features.add(dts);
					}

					for(Double dts : data[j].labels){
						testData[i].labels.add(dts);
					}
				}
				else{

					for(List<Double> dts : data[j].features){
						traineeData[i].features.add(dts);
					}

					for(Double dts : data[j].labels){
						traineeData[i].labels.add(dts);
					}
				}
			}
		}
		//end of making k-folds trainee and test data sets

		/*for(int i = 0; i < folds; i++){
			System.out.println(traineeData[i].features.size() + "\t" + traineeData[i].labels.size());
			System.out.println(testData[i].features.size() + "\t" + testData[i].labels.size());
		}*/

		int totalTestData = 0;
		int correctAnswer = 0;

		for (int j = 0; j < folds; j++) {

			System.out.println("Forest " + j + " is building...");

			Forest forest = new Forest();

			forest.makeDecisionTree(traineeData[j]);

			for (int i = 0; i < testData[j].features.size(); i++) {

				totalTestData++;
				
				double key = testData[j].labels.get(i);

				double returnedKey = forest.traverseForest(testData[j].features.get(i));

				//System.out.println("dataset = " + testData.attribute.get(i) + "\nActual decision = " + key
				//	+ "\nReturned decision = " + returnedKey);

				if (key == returnedKey) {
					//System.out.println("\nOriginal class: " + key + "\tPredicted class: " + returnedKey + "\tRight Prediction.");
					correctAnswer++;
				}
				else {
					//System.out.println("Original class: " + key + "\tPredicted class: " + returnedKey + "\tOOPS! Wrong Prediction.");
				}
			}
		}

		double accuracyRate = ((double)correctAnswer / ((double)totalTestData)) * 100.0;
		System.out.println("\nTotal test Data = " + totalTestData + ",\tCorrect Answer = " + correctAnswer);
		System.out.println("\nAccuracy Rate = " + accuracyRate + "%");
	}
}