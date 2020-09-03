 package ca.pfv.spmf.algorithms.frequentpatterns.fpgrowth;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



public class AlgoFPGrowth {

	public int minSuppRel;// the relative minimum support
	
	public int transCount = 0; // transaction count in the database
	
	public int itemSetC; // number of freq. itemsets found
	
	
	BufferedWriter writer = null; // object to write the output file
		
	// This variable is used to determine the size of buffers to store itemsets.
	// A value of 50 is enough because it allows up to 2^50 patterns!
	final int b_Size = 60;
	
	// buffer for storing the current itemset that is mined when performing mining
	// the idea is to always reuse the same buffer to reduce memory usage.
	public int[] itemsetBuffer = null;
	// another buffer for storing fpnodes in a single path of the tree
	public FPNode[] fpNodeTempBuffer = null;
	
	// This buffer is used to store an itemset that will be written to file
	// so that the algorithm can sort the itemset before it is output to file
	// (when the user choose to output result to file).
	public int[] itemsetOutputBuffer = null;
	
	/** maximum pattern length */
	public int maxPatternLength = 1000;

	/**
	 * Constructor
	 */
	public AlgoFPGrowth() {
		
	}

	/**
	 * Method to run the FPGRowth algorithm.
	 * @param input the path to an input file containing a transaction database.
	 * @param output the output file path for saving the result (if null, the result 
	 *        will be returned by the method instead of being saved).
	 * @param minsupp the minimum support threshold.
	 * @return the result if no output file path is provided.
	 * @throws IOException exception if error reading or writing files
	 */
	public void runAlgorithm(String input, String output, double minsupp) throws FileNotFoundException, IOException {
		
		// number of itemsets found
		itemSetC = 0;
		
		// if the user want to save the result to a file
		writer = new BufferedWriter(new FileWriter(output)); 
		itemsetOutputBuffer = new int[b_Size];
		
		
		// (1) PREPROCESSING: Initial database scan to determine the frequency of each item
		// The frequency is stored in a map:
		//    key: item   value: support
		final Map<Integer, Integer> mapSupport = scanDatabaseToDetermineFrequencyOfSingleItems(input); 


		/*
		 * 
		 * 
		 * done
		 * 
		 * 
		 */
		
		// convert the minimum support as percentage to a
		// relative minimum support
		this.minSuppRel = (int) Math.ceil(minsupp * transCount);
		
		// (2) Scan the database again to build the initial FP-Tree
		// Before inserting a transaction in the FPTree, we sort the items
		// by descending order of support.  We ignore items that
		// do not have the minimum support.
		FPTree tree = new FPTree();
		
		// read the file
		BufferedReader reader = new BufferedReader(new FileReader(input));
		String line;
		// for each line (transaction) until the end of the file
		while( ((line = reader.readLine())!= null)){ 
		
			
			String[] lineSplited = line.split(" ");
//			Set<Integer> alreadySeen = new HashSet<Integer>();
			List<Integer> transaction = new ArrayList<Integer>();
			
			// for each item in the transaction
			for(int i = 0; i < lineSplited.length; i++){  
				Integer item = Integer.parseInt(lineSplited[i]);
				// only add items that have the minimum support
				if(mapSupport.get(item) >= minSuppRel){
					transaction.add(item);	
				}
			}
			// sort item in the transaction by descending order of support
			Collections.sort(transaction, new Comparator<Integer>(){
				public int compare(Integer item1, Integer item2){
					// compare the frequency
					int compare = mapSupport.get(item2) - mapSupport.get(item1);
					// if the same frequency, we check the lexical ordering!
					return (compare == 0) ? (item1 - item2) : compare;
				}
			});
			
			
			// add the sorted transaction to the fptree.
			tree.addTransaction(transaction);
		}
		// close the input file
		reader.close();
		
		// We create the header table for the tree using the calculated support of single items
		tree.createHeaderList(mapSupport);
		
		// (5) We start to mine the FP-Tree by calling the recursive method.
		// Initially, the prefix alpha is empty.
		// if at least an item is frequent
		if(tree.headerList.size() > 0) {
			// initialize the buffer for storing the current itemset
			itemsetBuffer = new int[b_Size];
			// and another buffer
			fpNodeTempBuffer = new FPNode[b_Size];
			// recursively generate frequent itemsets using the fp-tree
			// Note: we assume that the initial FP-Tree has more than one path
			// which should generally be the case.
			fpgrowth(tree, itemsetBuffer, 0, transCount, mapSupport);
		}
		
		// close the output file if the result was saved to a file
		writer.close();
		
	}
	/**
	 * Mine an FP-Tree having more than one path.
	 * @param tree  the FP-tree
	 * @param prefix  the current prefix, named "alpha"
	 * @param mapSupport the frequency of items in the FP-Tree
	 * @throws IOException  exception if error writing the output file
	 */
	public void fpgrowth(FPTree tree, int [] prefix, int prefixLength, int prefixSupport, Map<Integer, Integer> mapSupport) throws IOException {
		
		if(prefixLength == maxPatternLength){
			return;
		}
		// We will check if the FPtree contains a single path
		boolean singlePath = true;
		// This variable is used to count the number of items in the single path
		// if there is one
		int position = 0;
		// if the root has more than one child, than it is not a single path
		if(tree.root.childs.size() > 1) {
			singlePath = false;
		}else {
			
			// Otherwise,
			// if the root has exactly one child, we need to recursively check childs
			// of the child to see if they also have one child
			FPNode currentNode = tree.root.childs.get(0);
			while(true){
				// if the current child has more than one child, it is not a single path!
				if(currentNode.childs.size() > 1) {
					singlePath = false;
					break;
				}
				// otherwise, we copy the current item in the buffer and move to the child
				// the buffer will be used to store all items in the path
				fpNodeTempBuffer[position] = currentNode;
				
				position++;
				// if this node has no child, that means that this is the end of this path
				// and it is a single path, so we break
				if(currentNode.childs.size() == 0) {
					break;
				}
				currentNode = currentNode.childs.get(0);
			}
		}
		
		// Case 1: the FPtree contains a single path
		if(singlePath){	
			// We save the path, because it is a maximal itemset
			saveAllCombinationsOfPrefixPath(fpNodeTempBuffer, position, prefix, prefixLength);
		}else {
			// For each frequent item in the header table list of the tree in reverse order.
			for(int i = tree.headerList.size()-1; i>=0; i--){
				// get the item
				Integer item = tree.headerList.get(i);
				
				// get the item support
				int support = mapSupport.get(item);
	
				// Create Beta by concatening prefix Alpha by adding the current item to alpha
				prefix[prefixLength] = item;
				
				// calculate the support of the new prefix beta
				int betaSupport = (prefixSupport < support) ? prefixSupport: support;
				
				// save beta to the output file
				saveItemset(prefix, prefixLength+1, betaSupport);
				
				if(prefixLength+1 < maxPatternLength){
				
					// === (A) Construct beta's conditional pattern base ===
					// It is a subdatabase which consists of the set of prefix paths
					// in the FP-tree co-occuring with the prefix pattern.
					List<List<FPNode>> prefixPaths = new ArrayList<List<FPNode>>();
					FPNode path = tree.mapItemNodes.get(item);
					
					// Map to count the support of items in the conditional prefix tree
					// Key: item   Value: support
					Map<Integer, Integer> mapSupportBeta = new HashMap<Integer, Integer>();
					
					while(path != null){
						// if the path is not just the root node
						if(path.parent.itemID != -1){
							// create the prefixpath
							List<FPNode> prefixPath = new ArrayList<FPNode>();
							// add this node.
							prefixPath.add(path);   // NOTE: we add it just to keep its support,
							// actually it should not be part of the prefixPath
							
							// ####
							int pathCount = path.counter;
							
							//Recursively add all the parents of this node.
							FPNode parent = path.parent;
							while(parent.itemID != -1){
								prefixPath.add(parent);
								
								// FOR EACH PATTERN WE ALSO UPDATE THE ITEM SUPPORT AT THE SAME TIME
								// if the first time we see that node id
								if(mapSupportBeta.get(parent.itemID) == null){
									// just add the path count
									mapSupportBeta.put(parent.itemID, pathCount);
								}else{
									// otherwise, make the sum with the value already stored
									mapSupportBeta.put(parent.itemID, mapSupportBeta.get(parent.itemID) + pathCount);
								}
								parent = parent.parent;
							}
							// add the path to the list of prefixpaths
							prefixPaths.add(prefixPath);
						}
						// We will look for the next prefixpath
						path = path.nodeLink;
					}
	
					// (B) Construct beta's conditional FP-Tree
					// Create the tree.
					FPTree treeBeta = new FPTree();
					// Add each prefixpath in the FP-tree.
					for(List<FPNode> prefixPath : prefixPaths){
						treeBeta.addPrefixPath(prefixPath, mapSupportBeta, minSuppRel); 
					}  
					
					// Mine recursively the Beta tree if the root has child(s)
					if(treeBeta.root.childs.size() > 0){
	
						// Create the header list.
						treeBeta.createHeaderList(mapSupportBeta); 
						// re'cursive call
						fpgrowth(treeBeta, prefix, prefixLength+1, betaSupport, mapSupportBeta);
					}
				}
			}
		}
		
	}


	/**
	 * This method saves all combinations of a prefix path if it has enough support
	 * @param prefix the current prefix
	 * @param prefixLength the current prefix length
	 * @param prefixPath the prefix path
	 * @throws IOException if exception while writting to output file
	 */
	public void saveAllCombinationsOfPrefixPath(FPNode[] fpNodeTempBuffer, int position, 
			int[] prefix, int prefixLength) throws IOException {

		int support = 0;
		// Generate all subsets of the prefixPath except the empty set
		// and output them
		// We use bits to generate all subsets.
loop1:	for (long i = 1, max = 1 << position; i < max; i++) {
			
			// we create a new subset
			int newPrefixLength = prefixLength;
			
			// for each bit
			for (int j = 0; j < position; j++) {
				// check if the j bit is set to 1
				int isSet = (int) i & (1 << j);
				// if yes, add the bit position as an item to the new subset
				if (isSet > 0) {
					if(newPrefixLength == maxPatternLength){
						continue loop1;
					}
					
					prefix[newPrefixLength++] = fpNodeTempBuffer[j].itemID;
					support = fpNodeTempBuffer[j].counter;
				}
			}
			// save the itemset
			saveItemset(prefix, newPrefixLength, support);
		}
	}
	

	/**
	 * This method scans the input database to calculate the support of single items
	 * @param input the path of the input file
	 * @throws IOException  exception if error while writing the file
	 * @return a map for storing the support of each item (key: item, value: support)
	 */
	public  Map<Integer, Integer> scanDatabaseToDetermineFrequencyOfSingleItems(String input)
			throws FileNotFoundException, IOException {
		
		// a map for storing the support of each item (key: item, value: support)
		 Map<Integer, Integer> mapSupport = new HashMap<Integer, Integer>();
		
		 //Create object for reading the input file
		BufferedReader reader = new BufferedReader(new FileReader(input));
		
		String line;
		// for each line (transaction) until the end of file
		while( ((line = reader.readLine())!= null)){ 
			
			// split the line into items
			String[] lineSplited = line.split(" ");
			// for each item
			for(String itemString : lineSplited){  
				// increase the support count of the item
				Integer item = Integer.parseInt(itemString);
				// increase the support count of the item
				Integer count = mapSupport.get(item);
				if(count == null){
					mapSupport.put(item, 1);
				}else{
					mapSupport.put(item, ++count);
				}
			}
			// increase the transaction count
			transCount++;
		}
		// close the input file
		reader.close();
		
		return mapSupport;
	}


	/**
	 * Write a frequent itemset that is found to the output file or
	 * keep into memory if the user prefer that the result be saved into memory.
	 */
	public void saveItemset(int [] itemset, int itemsetLength, int support) throws IOException {
		
		// increase the number of itemsets found for statistics purpose
		itemSetC++;
		
		// if the result should be saved to a file
			// copy the itemset in the output buffer and sort items
			System.arraycopy(itemset, 0, itemsetOutputBuffer, 0, itemsetLength);
			Arrays.sort(itemsetOutputBuffer, 0, itemsetLength);
			
			// Create a string buffer
			StringBuilder buffer = new StringBuilder();
			// write the items of the itemset
			for(int i=0; i< itemsetLength; i++){
				buffer.append(itemsetOutputBuffer[i]);
				if(i != itemsetLength-1){
					buffer.append(' ');
				}
			}
			// Then, write the support
			buffer.append(" #SUP: " + support);
			
			writer.write(buffer.toString());
			writer.newLine();	
		
	}
}

