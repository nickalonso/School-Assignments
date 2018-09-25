#Incorrect change assignment 

import java.util.Scanner;

public class MaxOverpayment {
    public static int payCalc(int bill, int payment) {
        int change;
        change=payment-bill;
                
        String temp=Integer.toString(change);
        int[] newChange=new int[temp.length()];
            for(int i=0; i<temp.length(); i++){
                newChange[i]=temp.charAt(i)-'0';}
                    
        int n=newChange.length;
            for(int i=0; i<n; i++){
                if(newChange[i]<9){
                   newChange[i]=9;
            break;}
        }
                
        int nums[]=newChange;
        StringBuilder strNum = new StringBuilder();
        
        for(int num: nums){
            strNum.append(num);}
        
        int finalInt=Integer.parseInt(strNum.toString());     
        int maxOverpay;
        maxOverpay=finalInt-change;
        System.out.print
        ("\nThe maximum amount Jim could overpay is $" +maxOverpay+"\n");
                        
        return change;
    }
         
    public static void main(String[] args) {
        Scanner A=new Scanner(System.in);
        System.out.println("Enter the billed amount: ");
 
        int a=A.nextInt();
        System.out.println("Enter the amount paid : ");  
                
        Scanner B=new Scanner(System.in);
        int b=B.nextInt();
        int change=payCalc(a,b);
    }
}
