package ca.pfv.spmf.patterns.itemset_array_integers_with_count;
import java.util.Arrays;
import java.util.List;

public class Itemset{
	/** the array of items **/
	public int[] itemset; 

	/**  the support of this itemset */
	public int support = 0; 
	
	/**
	 * Get the items as array
	 * @return the items
	 */
	public int[] getItems() {
		return itemset;
	}
	
	/**
	 * Constructor 
	 * @param item an item that should be added to the new itemset
	 */
	public Itemset(int item){
		itemset = new int[]{item};
	}

	/**
	 * Constructor 
	 * @param items an array of items that should be added to the new itemset
	 */
	public Itemset(int [] items){
		this.itemset = items;
	}
	
	
	/**
	 * Get the support of this itemset
	 */
	public int getAbsoluteSupport(){
		return support;
	}
	
	/**
	 * Get the size of this itemset 
	 */
	public int size() {
		return itemset.length;
	}

	/**
	 * Get the item at a given position in this itemset
	 */
	public Integer get(int position) {
		return itemset[position];
	}

	/**
	 * Set the support of this itemset
	 * @param support the support
	 */
	public void setAbsoluteSupport(Integer support) {
		this.support = support;
	}


	public void print() {
		System.out.print(toString());
	}
	public String toString(){
		if(size() == 0) {
			return "EMPTYSET";
		}
		// use a string buffer for more efficiency
		StringBuilder r = new StringBuilder ();
		// for each item, append it to the StringBuilder
		for(int i=0; i< size(); i++){
			r.append(get(i));
			r.append(' ');
		}
		return r.toString(); // return the tring
	}

	private boolean contains(int i) {
		// TODO Auto-generated method stub
		return false;
	}
}
