#linear
python linear.py ../data/2d_linear.npz --method default
python linear.py ../data/2d_linear.npz --method pinv
python linear.py ../data/2d_linear.npz --method qr
python linear.py ../data/2d_linear.npz --method lu
python linear.py ../data/2d_linear.npz --method qr_colamd
python linear.py ../data/2d_linear.npz --method lu_colamd

#loop
python linear.py ../data/2d_linear_loop.npz --method default
python linear.py ../data/2d_linear_loop.npz --method pinv
python linear.py ../data/2d_linear_loop.npz --method qr
python linear.py ../data/2d_linear_loop.npz --method lu
python linear.py ../data/2d_linear_loop.npz --method qr_colamd
python linear.py ../data/2d_linear_loop.npz --method lu_colamd

#non-linear
python nonlinear.py ../data/2d_nonlinear.npz --method default
python nonlinear.py ../data/2d_nonlinear.npz --method pinv
#python nonlinear.py ../data/2d_nonlinear.npz --method qr
python nonlinear.py ../data/2d_nonlinear.npz --method lu
#python nonlinear.py ../data/2d_nonlinear.npz --method qr_colamd
python nonlinear.py ../data/2d_nonlinear.npz --method lu_colamd