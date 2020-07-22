TimHanewich.NeuralNetwork
=========
Neural network system built on .NET for embedding machine learning (AI) in any project
--------

To download the latest version of this NuGet package in a dotnet package, run the following command with the .NET CLI:

    dotnet add package TimHanewich.NeuralNetwork

In this example, we will train a new neural network to model a NAND gate (https://en.wikipedia.org/wiki/NAND_gate).  
We use the following class to create a NAND gate example. Copy and paste this into your coding environment to use it in this NAND training example.

    public class Nand
    {
        public float Input1 {get; set;}
        public float Input2 {get; set;}
        public float Output {get; set;}

        public static Nand NewRandom()
        {
            Nand ToReturn = new Nand();
            Random r = new Random();
            int ir = r.Next(0, 4);

            if (ir == 0)
            {
                ToReturn.Input1 = 0;
                ToReturn.Input2 = 0;
                ToReturn.Output = 1;
            }
            else if (ir == 1)
            {
                ToReturn.Input1 = 0;
                ToReturn.Input2 = 1;
                ToReturn.Output = 1;
            }
            else if (ir == 2)
            {
                ToReturn.Input1 = 1;
                ToReturn.Input2 = 0;
                ToReturn.Output = 1;
            }
            else if (ir == 3)
            {
                ToReturn.Input1 = 1;
                ToReturn.Input2 = 1;
                ToReturn.Output = 0;
            }

            return ToReturn;
        }
    }

Place this statement at the top of your code to import the necessary resources:

    using TimHanewich.NeuralNetwork;

To create a new Neural Network:

    NeuralNetwork nn = NeuralNetwork.Create(new int[] {2, 3, 1}, true);

In the above code snippet, the first parameter, an `int` array, defines the structure of the network; the first value in this array is the number of input values, the last is the number of output values, and the middle values define how many hidden layers there should be and how many neurons should exist in each.
The more complex the scenario, the more hidden layers are needed. A NAND gate is a simple model, so we only need one hidden layer with three neurons in it.  

### Training the Neural Network
    for (int t = 0; t < 1000; t++)
    {
        Nand n = Nand.NewRandom();
        nn.ForwardPropagate(new float[] {n.Input1, n.Input2});
        nn.BackwardPropagate(new float[] {n.Output}, 0.3f);
    }
The `ForwardPropagate` method passes inputs through the model and returns the neural network's best guess at the correct output. Since we are only training the network, we can ignore the returned guess for now.  

The `BackwardPropagate` method is the process of "training" our network. This method accepts two parameters: the ideal output(s) (an array), and the learning rate, how strongly the model should adjust during this particular training example. The value of 0.3 is slightly high for typical scenarios, but appropriate as this is a training set of only 1,000 examples.

### Testing the Neural Network
    for (int t = 0; t < 1000; t++)
    {
        Nand n = Nand.NewRandom();
        float guess = nn.ForwardPropagate(new float[] {n.Input1, n.Input2})[0];
        Console.Write("Ideal: " + n.Output.ToString() + "    Guess: " + Math.Abs(guess).ToString("#,##0"));
        Console.ReadLine();
    }
In the above snippet, we are using the same `ForwardPropagate` method as before, but this time we are capturing the network's prediction in the `guess` variable. We then write the ideal (correct) output and the guess value to the console.  

The results of the above code:

    Ideal: 0    Guess: 0
    Ideal: 1    Guess: 1
    Ideal: 1    Guess: 1
    Ideal: 0    Guess: 0
    Ideal: 0    Guess: 0
    Ideal: 1    Guess: 1
    Ideal: 1    Guess: 1
    Ideal: 0    Guess: 0
    Ideal: 1    Guess: 1
    Ideal: 0    Guess: 0
    Ideal: 0    Guess: 0
    Ideal: 1    Guess: 1
