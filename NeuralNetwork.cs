using System;
using System.Collections.Generic;
using System.Linq;

namespace TimHanewich.NeuralNetwork
{
    public class NeuralNetwork
        {
            public List<Neuron> Neurons { get; set; }
            public List<Dendrite> Dendrites { get; set; }

            public NeuralNetwork()
            {
                Neurons = new List<Neuron>();
                Dendrites = new List<Dendrite>();
            }

            public Neuron CreateNeuron()
            {
                Neuron n = new Neuron();
                n.Id = Guid.NewGuid().ToString();
                n.InputValue = 0;
                n.IsBiasNeuron = false;
                Neurons.Add(n);
                return n;
            }

            public Dendrite CreateDendrite()
            {
                Dendrite d = new Dendrite();
                d.Id = Guid.NewGuid().ToString();
                d.Weight = 0f;
                d.NewWeight = null;
                Dendrites.Add(d);
                return d;
            }

            public static NeuralNetwork Create(int[] layer_definition, bool include_bias_neurons)
            {
                NeuralNetwork nn = new NeuralNetwork();



                List<Neuron> PreviousLayerNeurons = new List<Neuron>();
                for (int t = 1; t <= layer_definition.Length; t++)
                {
                    //Create this layers neurons
                    List<Neuron> ThisLayerNeurons = new List<Neuron>();
                    int NumberOfNeuronsForThisLayer = layer_definition[t - 1];
                    for (int nc = 1; nc <= NumberOfNeuronsForThisLayer; nc++)
                    {
                        Neuron n = nn.CreateNeuron();
                        ThisLayerNeurons.Add(n);
                    }



                    //If there are neurons in the "PreviousLayerNeurons" then connect the new ones to those
                    if (PreviousLayerNeurons.Count > 0)
                    {
                        foreach (Neuron tn in ThisLayerNeurons)
                        {
                            foreach (Neuron pn in PreviousLayerNeurons)
                            {
                                Dendrite d = nn.CreateDendrite();
                                d.InputNeuronId = pn.Id;
                                d.OutputNeuronId = tn.Id;
                            }
                        }
                    }


                    //Reset for next round
                    PreviousLayerNeurons.Clear();
                    PreviousLayerNeurons.AddRange(ThisLayerNeurons);
                    ThisLayerNeurons.Clear();


                    //Add a bias neuron if asked to (this has to be done after the full connections have been made in the previous step.
                    if (include_bias_neurons == true)
                    {
                        if (t != layer_definition.Length) //If it is not the last layer
                        {
                            Neuron bn = nn.CreateNeuron();
                            bn.InputValue = 1;
                            bn.IsBiasNeuron = true;
                            PreviousLayerNeurons.Add(bn);
                        }
                    }

                }


                //Randomize all weights
                nn.RandomizeAllWeights();

                return nn;
            }

            public Neuron[] GetInputNeurons()
            {
                List<Neuron> input_neurons = new List<Neuron>();
                foreach (Neuron n in Neurons)
                {
                    if (n.GetInputDendrites(this).Length == 0 && n.IsBiasNeuron == false)
                    {
                        input_neurons.Add(n);
                    }
                }
                return input_neurons.ToArray();
            }

            public Neuron[] GetOutputNeurons()
            {
                List<Neuron> output_neurons = new List<Neuron>();
                foreach (Neuron n in Neurons)
                {
                    if (n.GetOutputDendrites(this).Length == 0)
                    {
                        output_neurons.Add(n);
                    }
                }
                return output_neurons.ToArray();
            }

            public void RandomizeAllWeights()
            {
                Random r = new Random();
                foreach (Dendrite d in Dendrites)
                {
                    d.Weight = (float)r.NextDouble();
                }
            }

            public string Print()
            {
                List<string> All = new List<string>();

                //Print neurons
                All.Add("-----Neurons-----");
                for (int t = 0; t <= Neurons.Count - 1; t++)
                {
                    All.Add("[Neuron " + t.ToString("#,##0") + " (" + Neurons[t].Id + ")]" + " InputDendriteCount{" + Neurons[t].GetInputDendrites(this).Count().ToString() + "}");
                }

                //Print dendrites
                All.Add("-----Dendrites-----");
                for (int t = 0; t <= Dendrites.Count - 1; t++)
                {
                    All.Add("[Dendrite " + t.ToString("###0") + "]" + " Weight{" + Dendrites[t].Weight.ToString("#,##0.00") + "}");
                    All.Add("\t" + "InputNeuron{" + Dendrites[t].InputNeuronId + "}");
                    All.Add("\t" + "OutputNeuron{" + Dendrites[t].OutputNeuronId + "}");
                }



                string a = "";
                foreach (string s in All)
                {
                    a = a + s + Environment.NewLine;
                }
                a = a.Substring(0, a.Length - 1);

                return a;

            }

            private void SetIdealOutputs(float[] ideal_outputs)
            {
                Neuron[] outputns = GetOutputNeurons();
                if (ideal_outputs.Length != outputns.Length)
                {
                    throw new Exception("The number of ideal outputs supplied does not match the number of output neurons in the network.");
                }

                for (int t = 0; t <= ideal_outputs.Length - 1; t++)
                {
                    outputns[t].IdealValue = ideal_outputs[t];
                }
            }

            public float CalculateTotalError(float[] ideal_outputs)
            {
                Neuron[] outputns = GetOutputNeurons();
                if (ideal_outputs.Length != outputns.Length)
                {
                    throw new Exception("The number of ideal outputs supplied does not match the number of output neurons in the network.");
                }

                //Set the ideal outputs
                SetIdealOutputs(ideal_outputs);

                List<float> TotalError = new List<float>();
                for (int t = 0; t <= ideal_outputs.Length - 1; t++)
                {
                    TotalError.Add(outputns[t].CalculateError(this));
                }

                return TotalError.Sum();
            }


            //USAGE BELOW!
            public float[] ForwardPropagate(float[] inputs)
            {
                //Send in the inputs
                Neuron[] InputNeurons = GetInputNeurons();
                if (InputNeurons.Length != inputs.Length)
                {
                    throw new Exception("The number of inputs provided does not match the number of input neurons in the network.");
                }
                for (int t = 0; t <= inputs.Length - 1; t++)
                {
                    InputNeurons[t].InputValue = inputs[t];
                }

                Neuron[] OutputNeurons = GetOutputNeurons();
                List<float> Outputs = new List<float>();
                foreach (Neuron on in OutputNeurons)
                {
                    Outputs.Add(on.CalculateValue(this));
                }
                return Outputs.ToArray();
            }

            /// <summary>
            /// Backproagate the neural network to adjust dendrite weights.  Must have forward propagated the input value before using this.
            /// </summary>
            /// <param name="ideal_outputs">The ideal (correct outputs)</param>
            /// <param name="learning_rate">The applied learning rate during backpropagation</param>
            public void BackwardPropagate(float[] ideal_outputs, float learning_rate)
            {
                string thisbackpropid = Guid.NewGuid().ToString();

                //Set the ideal outputs
                SetIdealOutputs(ideal_outputs);

                //Get the new weights for all dendrites
                foreach (Dendrite d in Dendrites)
                {
                    Neuron InputtingNeuron = d.GetInputNeuron(this);
                    Neuron OutputtingNeuron = d.GetOutputNeuron(this);

                    if (OutputtingNeuron.GetOutputDendrites(this).Length == 0) //This is a dendrite that is connected directly to an output neuron.
                    {
                        float Gamma = OutputtingNeuron.CalculateGamma(this, thisbackpropid); //In the matt mazure example, gamma IS part 1 and 2 multiplied!  //The derivative of the activation function is INSIDE this function.
                        float Part3 = InputtingNeuron.CalculateValue(this, thisbackpropid);
                        float AllTogether = Gamma * Part3;
                        d.NewWeight = d.Weight - (learning_rate * AllTogether);
                    }
                    else //This is a dendrite that is outputting to a neuron that indirectly affects all output neurons.  Therefore the formula must be a little bit different.
                    {
                        float GammaOfOutputNeuron = OutputtingNeuron.CalculateGamma(this, thisbackpropid);
                        float ValueOfMyInputNeuron = InputtingNeuron.CalculateValue(this, thisbackpropid);
                        float AllTogether = GammaOfOutputNeuron * ValueOfMyInputNeuron;
                        d.NewWeight = d.Weight - (learning_rate * AllTogether);
                    }
                }

                //Update the weights of all dendrites
                foreach (Dendrite d in Dendrites)
                {
                    d.Weight = (float)d.NewWeight;
                    d.NewWeight = null;
                }
            }
        }

}