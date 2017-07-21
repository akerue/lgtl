# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class BD_LSTM(chainer.Chain):

    def __init__(self, vocab_size, word_dim, state_dim, initialW, EOS_ID):
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.state_dim = state_dim
        # self.initialW = initialW
        self.EOS_ID = EOS_ID
        
        if initialW is not None:
            assert initialW.shape[0] == vocab_size
            assert initialW.shape[1] == word_dim
            tmp = np.random.RandomState(1234).uniform(-0.01, 0.01, (vocab_size+1, word_dim))
            tmp[0:-1, :] = initialW
            initialW = tmp
        else:
            initialW = None
        self.vocab_size_in = self.vocab_size + 1
        self.BOS_ID = self.vocab_size_in - 1

        super(BD_LSTM, self).__init__(
            embed=L.EmbedID(self.vocab_size_in, self.word_dim,
                                ignore_label=-1, initialW=initialW),

            W_upd_fwd=L.Linear(self.word_dim, 4 * self.state_dim),
            U_upd_fwd=L.Linear(self.state_dim, 4 * self.state_dim, nobias=True),

            W_upd_bwd=L.Linear(self.word_dim, 4 * self.state_dim),
            U_upd_bwd=L.Linear(self.state_dim, 4 * self.state_dim, nobias=True),
            
            W_out=L.Linear(2 * self.state_dim, self.vocab_size),
        )
        self.U_upd_fwd.W.data[self.state_dim*0:self.state_dim*1, :] = self.init_ortho(self.state_dim)
        self.U_upd_fwd.W.data[self.state_dim*1:self.state_dim*2, :] = self.init_ortho(self.state_dim)
        self.U_upd_fwd.W.data[self.state_dim*2:self.state_dim*3, :] = self.init_ortho(self.state_dim)
        self.U_upd_fwd.W.data[self.state_dim*3:self.state_dim*4, :] = self.init_ortho(self.state_dim)

        self.U_upd_bwd.W.data[self.state_dim*0:self.state_dim*1, :] = self.init_ortho(self.state_dim)
        self.U_upd_bwd.W.data[self.state_dim*1:self.state_dim*2, :] = self.init_ortho(self.state_dim)
        self.U_upd_bwd.W.data[self.state_dim*2:self.state_dim*3, :] = self.init_ortho(self.state_dim)
        self.U_upd_bwd.W.data[self.state_dim*3:self.state_dim*4, :] = self.init_ortho(self.state_dim)

    def init_ortho(self, dim):
        A = np.random.randn(dim, dim)
        U, S, V = np.linalg.svd(A)
        return U.astype(np.float32)


    def forward(self, xs, ms, train=False):
        xs_rev = xs[::-1]
        ms_rev = ms[::-1]

        N = xs[0].data.shape[0]

        state_fwd = {
            "h": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            "c": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            }
        state_bwd = {
            "h": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            "c": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train),
            }

        ys_fwd = []
        for x, m in zip(xs, ms):
            state_fwd = self.update_state_fwd(x, state_fwd, train=train)
            m_ext = F.broadcast_to(F.reshape(m, (N, 1)), (N, self.state_dim))
            state_fwd["h"] = m_ext * state_fwd["h"]
            state_fwd["c"] = m_ext * state_fwd["c"]

            # y = self.predict(state_fwd["h"], train=train)
            y = state_fwd["h"]
            ys_fwd.append(y)

        ys_bwd = []
        for x, m in zip(xs_rev, ms_rev):
            state_bwd = self.update_state_bwd(x, state_bwd, train=train)
            m_ext = F.broadcast_to(F.reshape(m, (N, 1)), (N, self.state_dim))
            state_bwd["h"] = m_ext * state_bwd["h"]
            state_bwd["c"] = m_ext * state_bwd["c"]

            # y = self.predict(state_bwd["h"], train=train)
            y = state_bwd["h"]
            ys_bwd.append(y)

        # TODO: 一括でpredictできるように行列変換を行う
        ys = [self.predict(F.concat([y_fwd, y_bwd], axis=1), train=train) for y_fwd, y_bwd in zip(ys_fwd, ys_bwd[::-1])]

        return ys


    def update_state_fwd(self, x, state, train):
        v = self.embed(x)
        h_in = self.W_upd_fwd(v) + self.U_upd_fwd(state["h"])
        c, h = F.lstm(state["c"], h_in)
        state = {"h": h, "c": c}
        return state

    
    def update_state_bwd(self, x, state, train):
        v = self.embed(x)
        h_in = self.W_upd_bwd(v) + self.U_upd_bwd(state["h"])
        c, h = F.lstm(state["c"], h_in)
        state = {"h": h, "c": c}
        return state

    
    def predict(self, s, train):
        return self.W_out(s)
