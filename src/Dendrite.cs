using System;

namespace TimHanewich.NeuralNetwork
{
    public class Dendrite : NeuralComponent
        {
            public float Weight { get; set; }
            public float? NewWeight { get; set; }
            public string InputNeuronId { get; set; }
            public string OutputNeuronId { get; set; }

            public Neuron GetInputNeuron(NeuralNetwork ParentNetwork)
            {
                Neuron n = null;
                foreach (Neuron nn in ParentNetwork.Neurons)
                {
                    if (nn.Id == InputNeuronId)
                    {
                        n = nn;
                    }
                }

                if (n == null)
                {
                    throw new Exception("Unable to find input neuron with Id '" + InputNeuronId + "'.");
                }

                return n;

            }

            public Neuron GetOutputNeuron(NeuralNetwork ParentNetwork)
            {
                Neuron n = null;
                foreach (Neuron nn in ParentNetwork.Neurons)
                {
                    if (nn.Id == OutputNeuronId)
                    {
                        n = nn;
                    }
                }

                if (n == null)
                {
                    throw new Exception("Unable to find output neuron with Id '" + InputNeuronId + "'.");
                }

                return n;

            }

        }
}