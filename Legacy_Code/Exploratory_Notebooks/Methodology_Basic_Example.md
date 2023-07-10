$$
\newcommand{\A}{
\begin{bmatrix}
Y(S^*) \\
Y(S) \\
Z(S')
\end{bmatrix}
}

\newcommand{\YI}{
\begin{bmatrix}
Y(S^*) 
\end{bmatrix}
}

\newcommand{\YII}{
\begin{bmatrix}
Y(S) \\
Z(S')
\end{bmatrix}
}

\newcommand{\YY}{
\begin{bmatrix}
Y_1 \\
Y_2
\end{bmatrix}
}

\newcommand{\U}{
\begin{bmatrix}
\mu_W(S^*) \\
\mu_W(S) \\
\mu_{W+D}(S')
\end{bmatrix}
}

\newcommand{\UI}{
\begin{bmatrix}
\mu_W(S^*) 
\end{bmatrix}
}

\newcommand{\UII}{
\begin{bmatrix}
\mu_W(S) \\
\mu_{W+D}(S')
\end{bmatrix}
}

\newcommand{\UU}{
\begin{bmatrix}
U_1 \\
U_2
\end{bmatrix}
}

\newcommand{\C}{
\begin{bmatrix}
K_W(S^*,S^*) & K_W(S^*,S) & K_W(S^*,S') \\
K_W(S,S^*) & K_W(S,S) & K_W(S,S') \\
K_W(S',S^*) & K_W(S',S) & K_W(S',S')+K_D(S',S')
\end{bmatrix}
}

\newcommand{\KI}{
\begin{bmatrix}
K_W(S^*,S^*) 
\end{bmatrix}
}

\newcommand{\KII}{
\begin{bmatrix}
K_W(S^*,S) & K_W(S^*,S') 
\end{bmatrix}
}

\newcommand{\KIII}{
\begin{bmatrix}
K_W(S,S^*) \\
K_W(S',S^*)
\end{bmatrix}
}

\newcommand{\KIV}{
\begin{bmatrix}
K_W(S,S) & K_W(S,S') \\
K_W(S',S) & K_W(S',S')+K_D(S',S')
\end{bmatrix}
}

\newcommand{\KK}{
\begin{bmatrix}
K_{11} & K_{12} \\
K_{21} & K_{22} 
\end{bmatrix}
}
$$

$$
\begin{align*}
  Y(S) &\sim W(S)\\
  Z(S) &\sim W(S)+D(S)\\
\end{align*}
$$

$$
\A \sim MVN\left(\U,\C\right)
$$
Let:
$$
Y_1 = \YI, Y_2=\YII
$$
$$
U_1 = \UI, U_2=\UII
$$
$$
K_{11} = \KI, K_{12} = \KII
$$
$$
K_{21} = \KIII, K_{22} = \KIV
$$

Then:
$$
\YY \sim MVN\left(\UU,\KK\right)
$$

and:
$$
P(Y_1|Y_2) = MVN(U_{1|2},K_{1|2})
$$

where:
$$
U_{1|2} = U_1 + K_{12}K_{22}^{-1}(Y_2-U_2)
$$
$$
K_{1|2} = K_{11} - K_{12}K_{22}^{-1}K_{21}
$$
$$
K_{22} = L_{22}L_{22}^T
$$
$$
K_{22}^{-1} = L_{22}^{-T}L_{22}^{-1}
$$
$$
K_{1|2} = K_{11} - K_{12}L_{22}^{-T}(L_{22}^{-1}K_{21})
$$
$$
P_{21} = L_{22}^{-1}K_{21}
$$
$$
K_{1|2} = K_{11} - P_{21}^TP_{21}

$$
<!-- Commands:
$$
A = \A,  B = \B, C=\C \\
$$
\\
$$
KI = \KI, KII = \KII
$$
$$
KI = \KIII, KII = \KIV
$$ -->