---
title: "Let's build a calculator(1) - Basic & Useless Calculator"
date: 2020-03-08
categories:
 - calculator 

tags:
 - calculator
 - stack
 - data structure
 - c++

---


The task is to devise a "basic" calculator.

Our calculator supports:
1) 1 digit operand
2) supporting basic arithmetic operations(+, -, *, /)

It seems quite **useless**, but let's do easier things first.

*C++* is used for representation. Let's start.

***

### Basic Setting

{% highlight cpp %}
#include <iostream>
#include <stdlib.h>
#include <string>
#include <stack>

using namespace std;
{% endhighlight %}


### Parser 

We should be able to evaluate, for example, **(1+2)*7** .

We certainly know that we should do **(1+2)** first, and **multiply 7** and so on ...

However, computer does not :(

That is the reason why we need to parse the formula, so that computer can understand what we meant to do.

We will turn [infix][infix] (human-friendly notation) into [postfix][postfix] first.
 
Here is the great idea to do so, from [geeksforgeeks][geeksforgeeks]: 

```
Algorithm
1. Scan the infix expression from left to right.
2. If the scanned character is an operand, output it.
3. Else,
…..3.1 If the precedence of the scanned operator is greater than the precedence of the operator in the stack(or the stack is empty or the stack contains a ‘(‘ ), push it.
…..3.2 Else, Pop all the operators from the stack which are greater than or equal to in precedence than that of the scanned operator. After doing that Push the scanned operator to the stack. (If you encounter parenthesis while popping then stop there and push the scanned operator in the stack.)
4. If the scanned character is an ‘(‘, push it to the stack.
5. If the scanned character is an ‘)’, pop the stack and and output it until a ‘(‘ is encountered, and discard both the parenthesis.
6. Repeat steps 2-6 until infix expression is scanned.
7. Print the output
8. Pop and output from the stack until it is not empty.
```
So for our formula, **(1+2)*7**, we have following procedure.

![inToPost](/assets/images/post-2020-03-08-inToPost.jpg)


Let's define our helper functions first.
{% highlight cpp %}

int opPriority(char c)
{
	switch(c)
	{
		case '(':
			return 0;
		case '+': case '-':
			return 1;
		case '*': case '/':
			return 2;
		case ')':
			return 100;
	}
	return 0;
}

double calculate(double a, double b, char op)
{
	switch(op)
	{
		case '+':
			return a + b;
		case '-':
			return a - b;
		case '*':
			return a * b;
		case '/':
			return a / b;
	}
}

{% endhighlight %}


Let's get into our parser design!

{% highlight cpp %}
string parseString(string s)
{
		/*
			String Parser
			classify cases by char
			0) blank : pass
			1) digit : just print out (to new string)
			2) op/ parenthesis : get priority.
				2-1) parenthesis '(' - just push, ')' - pop until '(' appears. 
				2-2) stack is empty or current op priority is bigger stack.top
				2-3) current op priority is smaller than stack.top
		*/
	
	string parsedString;
	stack<char> opStack; 


	for (string::iterator c = s.begin(); c != s.end(); ++c)
	{
		// pass the blank
		if(*c == ' ') continue;

		if(isdigit(*c)) // case 1
		{
			parsedString.push_back(*c);
		} 
		else // case 2
		{
			if(*c == '(')
			{
				opStack.push(*c);
			}
			else if(*c == ')')
			{
				while(opStack.top() != '(')
					{
						parsedString.push_back(opStack.top());
						opStack.pop();	
					}
					opStack.pop(); // pop '('
			}
			else if (opStack.empty() || 
					opPriority(opStack.top()) <= opPriority(*c) ) // case 2-1
			{
				opStack.push(*c);
			}
			else // case 2-2
			{
				parsedString.push_back(opStack.top());
				opStack.pop();	
				opStack.push(*c);
			}

		}
		
	}

	while(!opStack.empty())
	{
		parsedString.push_back(opStack.top());
		opStack.pop();
	}

	return parsedString;
}

{% endhighlight %}

So if input is **"(1+2)*7"**, 

output is **"12+7*"**.

### Postfix Notation Evaluator

We should actually be able to calcuate the result from postfix notation.

i.e. **"12+7*"** into 21.

Our idea is as follows:

![eval](/assets/images/post-2020-03-08-eval.jpg)

{% highlight cpp %}

double evaluatePostfix(string s)
{
	stack<double> op;
	double result;
	for (string::iterator c = s.begin(); c != s.end(); ++c)
	{
		if(isdigit(*c))
		{
			char value = *c;
			op.push(atof(&value));
		} 
		else 
		{
			double operand2 = (double)op.top();
			op.pop();
			double operand1 = (double)op.top();
			op.pop();
			double sub_result = calculate(operand1, operand2, *c);
			op.push(sub_result);
		}
	}
	
	result = op.top();
	op.pop(); // empty stack
	return result;
}


{% endhighlight %}

### Testing

{% highlight cpp %}

int main(void)
{
	string formula;
	string formula_postfix;
	double result;

	cout << "Input your formula to evalute." << endl;

	getline(cin, formula); 

	formula_postfix = parseString(formula);

	cout << formula_postfix << endl;
	result = evaluatePostfix(formula_postfix);
	
	cout << "Result: " << result << endl;
}
{% endhighlight %}

***

Compile & implement the code as:
```
g++ -o calc Calculator.cpp
./calc
```

Result is:
```
Input your formula to evalute.
(1+2)*7
12+7*
Result: 21
```

### OOP feature Added(2020-03-13)
We can organize the code by adding OOP features of C++.
Let's get into the code.

{% highlight cpp %}

/*
implementation of basic calculator using C++
OOP features
1 digit, basic arithmetic calcuations(+, -, *, /).

by Dongmin Kim(dongmin.kim.0220@gmail.com)
*/


#include <iostream>
#include <stdlib.h>
#include <string>
#include <stack>

using namespace std;

class Calculator{

private:
	string formula;	

public:

	Calculator() {
		cout << "Calculater Ver1!" << endl;
		
		while(1)
		{
			Calculator::formula = ""; // initialize
			cout << "Input your formula to evalute. Type 'exit' to terminate" << endl;
			getline(cin, Calculator::formula); 

			if(Calculator::formula == "EXIT" ||
				 Calculator::formula == "exit" ||
				 Calculator::formula == "Exit"
			) break;
			
			evaluate();
		}
		
		cout << "Program Ended" << endl;		
	}
 		
	double evaluate()
	{
		string formula_postfix = Calculator::parseString(Calculator::formula);
		double result = evaluatePostfix(formula_postfix);
		cout << "Result: " << result << endl;
		return result;
	}

	int opPriority(char c)
	{
		switch(c)
		{
			case '+': case '-':
				return 1;
			case '*': case '/':
				return 2;
		}
		return 0;
	}
	
	
	double calculate(double a, double b, char op)
	{
		switch(op)
		{
			case '+':
				return a + b;
			case '-':
				return a - b;
			case '*':
				return a * b;
			case '/':
				return a / b;
		}
	}

	string parseString(string s)
	{
		/*
			String Parser
			classify cases by char
			0) blank : pass
			1) digit : just print out (to new string)
			2) op/ parenthesis : get priority.
				2-1) parenthesis '(' - just push, ')' - pop until '(' appears. 
				2-2) stack is empty or current op priority is bigger stack.top
				2-3) current op priority is smaller than stack.top
		*/
	
		string parsedString;
		stack<char> opStack; 


		for (string::iterator c = s.begin(); c != s.end(); ++c)
		{
			// pass the blank
			if(*c == ' ') continue;

			if(isdigit(*c)) // case 1
			{
				parsedString.push_back(*c);
			} 
			else // case 2
			{
				if(*c == '(')
				{
					opStack.push(*c);
				}
				else if(*c == ')')
				{
					while(opStack.top() != '(')
						{
							parsedString.push_back(opStack.top());
							opStack.pop();	
						}
						opStack.pop(); // pop '('
				}
				else if (opStack.empty() || 
						opPriority(opStack.top()) <= opPriority(*c) ) // case 2-1
				{
					opStack.push(*c);
				}
				else // case 2-2
				{
					parsedString.push_back(opStack.top());
					opStack.pop();	
					opStack.push(*c);
				}
	
			}
			
		}

		while(!opStack.empty())
		{
			parsedString.push_back(opStack.top());
			opStack.pop();
		}

		return parsedString;
	}

	double evaluatePostfix(string s)
	{
		stack<double> op;
		double result;
		for (string::iterator c = s.begin(); c != s.end(); ++c)
		{
			if(isdigit(*c))
			{
				char value = *c;
				op.push(atof(&value));
			} 
			else 
			{
				double operand2 = (double)op.top();
				op.pop();
				double operand1 = (double)op.top();
				op.pop();
				double sub_result = calculate(operand1, operand2, *c);
				op.push(sub_result);
			}
		}
	
		result = op.top();
		op.pop(); // empty stack
		return result;
	}

};

int main(void)
{
	Calculator c = Calculator();
	
}


{% endhighlight %}

Let's see how this program works.

Compile & implement the code as:
```
g++ -o calc Calculator.cpp
./calc
```

```
Calculater Ver1!
Input your formula to evalute. Type 'exit' to terminate
1+2
Result: 3
Input your formula to evalute. Type 'exit' to terminate
2+3
Result: 5
Input your formula to evalute. Type 'exit' to terminate
(1+2)/3
Result: 1
Input your formula to evalute. Type 'exit' to terminate
1+2/3
Result: 1.66667
Input your formula to evalute. Type 'exit' to terminate
exit
Program Ended

```



Code is available at my [github][github].


[infix]: http://www.cs.man.ac.uk/~pjj/cs212/fix.html
[postfix]: http://www.cs.man.ac.uk/~pjj/cs212/fix.html
[geeksforgeeks]: https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
[github]: https://github.com/dongminkim0220/Calculator

