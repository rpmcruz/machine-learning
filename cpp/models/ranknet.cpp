// Basic neural network (uses sigmoid on all layers)

#include <iostream>
using namespace std;
#include <armadillo>
using namespace arma;

//** utilities

static mat sigmoid(const mat &x) {
    return 1/(1+exp(-x));
}

static mat dsigmoid(const mat &x) {
    // dsigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    return x%(1-x);
}

static void bincount(const ivec& y, int *zeros, int *ones) {
    *zeros = (*ones = 0);
    for(unsigned int i = 0; i < y.n_elem; i++) {
        if(y[i] == 1)
            (*ones)++;
        else //if(y[i] == 0)
            (*zeros)++;
    }
}

static void calccosts(const ivec& y, double *Cn, double *Cp) {
    int zeros, ones;
    bincount(y, &zeros, &ones);
    *Cn = y.n_elem / (2. * zeros);
    *Cp = y.n_elem / (2. * ones);
}

static double choose_threshold(vec s, ivec y) {
    uvec si = sort_index(s);
    s = s.rows(si);
    y = y.rows(si);

    double maxF1 = 0;
    double bestTh = 0;
    for(unsigned int i = 1; i < y.n_elem; i++)
        if(y[i] != y[i-1]) {
            int TP = 0, FN = 0, FP = 0;
            for(unsigned int j = 0; j < i; j++)
                if(y[j] == 1)
                    FN++;
            for(unsigned int j = i; j < y.n_elem; j++) {
                if(y[j] == 1)
                    TP++;
                else
                    FP++;
            }
            double F1 = (2.*TP)/(2.*TP+FN+FP);
            if(F1 > maxF1) {
                maxF1 = F1;
                bestTh = (s[i]+s[i-1])/2.;
            }
        }
    return bestTh;
}

//** implementations

struct NeuralNet
{
    int hidden_nodes;
    bool balanced;
    double th;

    mat w0, b0, w1, b1;
    rowvec Xmin, Xmax;

    static const double rang = 0.7;
    static const double eta = 1;

    NeuralNet(int hidden_nodes, bool balanced) {
        this->hidden_nodes = hidden_nodes;
        this->balanced = balanced;
        th = 0.5;
    }

    virtual ~NeuralNet() {}

    void build(int k0, int k1) {
        w0 = rang * (randu(k0, k1)*2-1);
        b0 = rang * (randu(1, k1)*2-1);
        w1 = rang * (randu(k1, 1)*2-1);
        b1 = rang * (randu(1,1)*2-1);
    }

    mat* fprop(const mat& X) {
        mat* l = new mat[3];
        l[0] = X;
        l[1] = sigmoid((l[0] * w0) + b0);  // (1,k1)
        l[2] = sigmoid((l[1] * w1) + b1);  // (1)
        return l;
    }

    void backprop(double C, int sign, const mat& l0, const mat& l1, const mat& l2) {
        // nodes are updated based on how much impact they have in the
        // next layer times their unconfidence (dsigmoid)
        mat l2_ = dsigmoid(l2);
        mat delta1 = C*l2_;

        mat l1_ = dsigmoid(l1);
        mat delta0 = delta1 * (w1.t() % l1_);  // (1,k1)

        b1 += eta * delta1 * sign;
        w1 += eta * (l1.t() * delta1) * sign;
        b0 += eta * delta0 * sign;
        w0 += eta * (l0.t() * delta0) * sign;
    }

    mat applynorm(mat X) {
        X.each_row() -= Xmin;
        X.each_row() /= Xmax - Xmin + 1e-6;
        return X;
    }

    mat fitnorm(const mat& X) {
        Xmin = min(X, 0);
        Xmax = max(X, 0);
        return applynorm(X);
    }

    virtual void fit(mat X, const ivec& y, int maxit) {
        build(X.n_cols, hidden_nodes);
        X = fitnorm(X);

        double costs[2] = { 1, 1 };
        if(balanced)
            calccosts(y, &costs[0], &costs[1]);
        for(int t = 0; t < maxit; t++) {
            double error = 0;
            for(unsigned int i = 0; i < X.n_rows; i++) {
                mat* l = fprop(X.row(i));

                double C = (l[2][0] - y[i]) * costs[(int)y[i]];
                backprop(C, -1, l[0], l[1], l[2]);
                error += abs(C);

                delete [] l;
            }
            //if(t % 100 == 0)
            //    cout << error/X.n_rows << endl;
            if(error/X.n_rows < 0.01)  // has converged
                break;
        }
    }

    void scores(mat X, bool normalize, vec& s) {
        if(normalize)
            X = applynorm(X);
        //vec s(X.n_rows);
        for(unsigned int i = 0; i < X.n_rows; i++)
            s[i] = fprop(X.row(i))[2][0];
        //return s;
    }

    void predict(mat X, ivec& yp) {
        X = applynorm(X);
        //uvec yp(X.n_rows);
        for(unsigned int i = 0; i < X.n_rows; i++)
            yp[i] = (fprop(X.row(i))[2][0] >= th) ? 1 : 0;
        //return yp;
    }
};

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
    NeuralNet* NeuralNet_new(int hidden_nodes, bool balanced) {
        return new NeuralNet(hidden_nodes, balanced);
    }
    void NeuralNet_delete(NeuralNet* nn) {
        delete nn;
    }
    void NeuralNet_fit(NeuralNet* nn, int D, int N, void* X, void* y, int maxit) {
        mat _X((double*) X, N, D, false, true);
        ivec _y((int*) y, N, false, true);
        nn->fit(_X, _y, maxit);
    }
    void NeuralNet_predict(NeuralNet* nn, int D, int N, void* X, void* yp) {
        mat _X((double*) X, N, D, false, true);
        ivec _yp((int*) yp, N, false, true);
        nn->predict(_X, _yp);
    }
    void NeuralNet_scores(NeuralNet* nn, int D, int N, void* X, void* S) {
        mat _X((double*) X, N, D, false, true);
        vec _S((double*) S, N, false, true);
        nn->scores(_X, true, _S);
    }

    NeuralNet* RankNet_new(int hidden_nodes, bool balanced) {
        return new RankNet(hidden_nodes);
    }
}
