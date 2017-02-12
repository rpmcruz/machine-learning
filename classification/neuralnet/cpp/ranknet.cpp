#include "neuralnet.cpp"

// RankNet as in Burges et al (2005)
// Uses a neural network for ranking.

struct RankNet : NeuralNet
{
    RankNet(int hidden_nodes) : NeuralNet(hidden_nodes, false) {
    }

    virtual void fit(mat X, const ivec& y, int maxit) {
        build(X.n_cols, hidden_nodes);
        X = fitnorm(X);

/*        int n1 = 0;
        for(unsigned int i = 0; i < y.n_elem; i++)
            if(y[i] == 1)
                n1++;
        unsigned int _maxit = maxit/n1;
        cout << "maxit: " << _maxit << endl;*/

        for(int t = 0; t < maxit; t++) {
            double errors = 0;
            for(unsigned int i = 0; i < X.n_rows; i++)
                for(unsigned int j = 0; j < X.n_rows; j++) {
                    if(i == j)
                        continue;
#if 1  // ignore same -- they only affect the threshold, not the learning
                    if(y[i] == y[j])
                        continue;
#endif
                    double P = (y[i] - y[j] + 1)/2.;

                    mat* l1 = fprop(X.row(i));
                    mat* l2 = fprop(X.row(j));

                    double s = l1[2][0] - l2[2][0];
                    double C = exp(s)/(exp(s)+1) - P;

                    backprop(C, -1, l1[0], l1[1], l1[2]);
                    backprop(C, +1, l2[0], l2[1], l2[2]);
                    errors += abs(C);
                    delete [] l1;
                    delete [] l2;
                }
            if(t % 100 == 0)
                cout << t << " - " << errors << endl;
            if(errors/(X.n_rows*(X.n_rows-1)) < 0.01)
                break;
        }

        vec H(X.n_rows);
        scores(X, false, H);
        th = choose_threshold(H, y);
    }
};

//** wrapper

extern "C" {
    NeuralNet* RankNet_new(int hidden_nodes, bool balanced) {
        return new RankNet(hidden_nodes);
    }
}
