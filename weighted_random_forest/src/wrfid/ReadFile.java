package wrfid;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ReadFile {

	public List<List<Double>> features = new ArrayList<>();
	public List<Double> labels = new ArrayList<>();
	public DataNode dtnd = new DataNode();

	public void fileReader(String filepath) {

		InputStream is = null;
		InputStreamReader isr = null;
		BufferedReader br = null;

		try {
			is = new FileInputStream(filepath);

			isr = new InputStreamReader(is);

			br = new BufferedReader(isr);

			String s = new String();

			if(br.readLine() == null){
				System.out.println("Empty file!");
				System.exit(0);
			}
			
			while ((s = br.readLine()) != null) {

				String[] separate = s.split(",");

				if(separate.length <= 1){
					System.out.println("Wrong Format! Data must be comma separated.");
					System.exit(0);
				}
				
				double decisionAttribute = Double.parseDouble(separate[0]);

				this.labels.add(decisionAttribute);

				List<Double> temp = new ArrayList<>();

				for (int i = 1; i < separate.length; i++) {

					double matrixElement = Double.parseDouble(separate[i]);
					temp.add(matrixElement);

				}

				this.features.add(temp);

			}

			ArrayList<Integer> indices = new ArrayList<Integer>();

			for (int i = 0; i < this.labels.size(); i++) {
				indices.add(i);
			}

			Collections.shuffle(indices);

			for (int i = 0; i < indices.size(); i++) {

				this.dtnd.features.add(this.features.get(indices.get(i)));
				this.dtnd.labels.add(this.labels.get(indices.get(i)));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	
	public DataNode[] makeFolds(int folds) {

		DataNode[] foldData = new DataNode[folds];

		int count = 0;
		int foldSize = (int) Math.ceil((double)this.dtnd.features.size()/folds);

		foldData[count] = new DataNode();

		for(int i = 0; i < this.dtnd.features.size(); i++){

			foldData[count].features.add(this.dtnd.features.get(i));
			foldData[count].labels.add(this.dtnd.labels.get(i));

			if((i+1)%foldSize == 0){
				count++;
				if(count != folds){
					foldData[count] = new DataNode();
				}
				else{
					count--;
				}
			}
			
		}

		return foldData;
	}
	
	
	public Map<Double, Double> assignClassWeights(){
		
		Map<Double, Double> classInstanceCount = new HashMap<>();
		Map<Double, Double> classWeight = new HashMap<>();
		for(int i = 0; i < this.labels.size(); i++){
			
			if(classInstanceCount.containsKey(this.labels.get(i))){
				classInstanceCount.put(this.labels.get(i), classInstanceCount.get(this.labels.get(i)) + 1.0);
			}
			else{
				classInstanceCount.put(this.labels.get(i), 1.0);
			}
		}
		
		ArrayList<Double> temp = new ArrayList<>();
		
		for(Double dt : classInstanceCount.keySet()){
			temp.add(classInstanceCount.get(dt));
		}
		
		Collections.sort(temp);
		
		int clscnt = classInstanceCount.size();
		
		for(int i = 0; i < clscnt; i++){
			
			for(Double dt : classInstanceCount.keySet()){
				if(temp.get(i)==classInstanceCount.get(dt)){
					classWeight.put(dt, 1.0 - (temp.get(i)/(double)this.labels.size()));
				}
			}
		}

		return classWeight;
		
	}

}
