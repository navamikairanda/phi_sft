/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "physics.hpp"

#include "collisionutil.hpp"
#include "taucs.hpp"

using namespace std;
using namespace torch;

static const bool verbose = false;

static const vector<Cloth::Material*> *materials;

template <Space s>
Tensor stretching_energy (const Face *face) {
    Tensor F = derivative(pos<s>(face->v[0]->node), pos<s>(face->v[1]->node),
                          pos<s>(face->v[2]->node), face);
    Tensor G = (matmul(F.t(),F) - torch::eye(2,TNOPT)).flatten()/2.;
    Tensor k = stretching_stiffness(G, (*::materials)[face->label]->stretching);
    Tensor weakening = (*::materials)[face->label]->weakening;
    k = k/(1 + weakening*face->damage);
    Tensor ans = face->a*(k[0]*sq(G[0]) + k[2]*sq(G[3])
                    + 2*k[1]*G[0]*G[3] + k[3]*sq(G[1]))/2.;
    return ans;
}

template <Space s>
pair<Tensor,Tensor> stretching_force (const Face *face) {
    Tensor F = derivative(pos<s>(face->v[0]->node), pos<s>(face->v[1]->node),
                          pos<s>(face->v[2]->node), face);
    Tensor G = (matmul(F.t(),F) - torch::eye(2,TNOPT)).flatten()*0.5;
    Tensor k = stretching_stiffness(G, (*::materials)[face->label]->stretching);
    Tensor weakening = (*::materials)[face->label]->weakening;
    k = k/(1 + weakening*face->damage);
    // eps = 1/2(F'F - I) = 1/2([x_u^2 & x_u x_v \\ x_u x_v & x_v^2] - I)
    // e = 1/2 k0 eps00^2 + k1 eps00 eps11 + 1/2 k2 eps11^2 + k3 eps01^2
    // grad e = k0 eps00 grad eps00 + ...
    //        = k0 eps00 Du' x_u + ...
    Tensor d = face->invDm.flatten();//acbd
    Tensor Du = cat({(-d[0]-d[2])*EYE3,d[0]*EYE3,d[2]*EYE3},1),//kronecker(rowmat(du), torch::eye(3,TNOPT)),
             Dv = cat({(-d[1]-d[3])*EYE3,d[1]*EYE3,d[3]*EYE3},1);//kronecker(rowmat(dv), torch::eye(3,TNOPT));
    const Tensor &xu = F.slice(1,0,1).squeeze(), &xv = F.slice(1,1,2).squeeze(); // should equal Du*mat_to_vec(X)

    Tensor Dut = Du.t(), Dvt = Dv.t();
    Tensor fuu = matmul(Dut,xu), fvv = matmul(Dvt,xv), fuv = matmul(Dut,xv) + matmul(Dvt,xu);
    Tensor grad_e = k[0]*G[0]*fuu + k[2]*G[3]*fvv
                + k[1]*(G[0]*fvv + G[3]*fuu) + k[3]*G[1]*fuv;

    G = relu(G);
    Tensor DutDu = matmul(Dut, Du), DvtDv = matmul(Dvt, Dv);
    Tensor hess_e = k[0]*(ger(fuu,fuu) + G[0]*DutDu)
                  + k[2]*(ger(fvv,fvv) + G[3]*DvtDv)
                  + k[1]*(ger(fuu,fvv) + G[0]*DvtDv
                          + ger(fvv,fuu) + G[3]*DutDu)
                  + 0.5*k[3]*(ger(fuv,fuv));
    // ignoring G[0][1]*(Du.t()*Dv+Dv.t()*Du)/2. term
    // because may not be positive definite
    return make_pair(-face->a*hess_e, -face->a*grad_e);
}

pair<Tensor,Tensor> batch_stretching_force (Tensor batch_F, Tensor batch_stret, Tensor batch_weak, Tensor batch_du, Tensor batch_dv, Tensor batch_a) {
    Tensor G = (bmm(batch_F.permute({0,2,1}),batch_F) - torch::eye(2,TNOPT).unsqueeze(0)).reshape({-1,4})*0.5;
    G = G.t();
    Tensor k = batch_stretching_stiffness(G, batch_stret);
    k = k/batch_weak;
    // eps = 1/2(F'F - I) = 1/2([x_u^2 & x_u x_v \\ x_u x_v & x_v^2] - I)
    // e = 1/2 k0 eps00^2 + k1 eps00 eps11 + 1/2 k2 eps11^2 + k3 eps01^2
    // grad e = k0 eps00 grad eps00 + ...
    //        = k0 eps00 Du' x_u + ...
    const Tensor &xu = batch_F.slice(2,0,1), &xv = batch_F.slice(2,1,2); // should equal Du*mat_to_vec(X)

    Tensor Dut = batch_du.permute({0,2,1}), Dvt = batch_dv.permute({0,2,1});
    Tensor fuu = bmm(Dut,xu).squeeze(), fvv = bmm(Dvt,xv).squeeze(), fuv = (bmm(Dut,xv) + bmm(Dvt,xu)).squeeze();//nx9
    Tensor grad_e = k[0]*G[0]*fuu.t() + k[2]*G[3]*fvv.t()
                + k[1]*(G[0]*fvv.t() + G[3]*fuu.t()) + k[3]*G[1]*fuv.t();
//cout << "stret" << endl;
//cout << G[0]*fuu.t() << endl;
//cout << k << endl;
    G = relu(G);
    Tensor DutDu = bmm(Dut, batch_du).permute({1,2,0}), DvtDv = bmm(Dvt, batch_dv).permute({1,2,0});//9x9xn
    Tensor hess_e = k[0]*(bmm(fuu.unsqueeze(2),fuu.unsqueeze(1)).permute({1,2,0}) + G[0]*DutDu)
                  + k[2]*(bmm(fvv.unsqueeze(2),fvv.unsqueeze(1)).permute({1,2,0}) + G[3]*DvtDv)
                  + k[1]*(bmm(fuu.unsqueeze(2),fvv.unsqueeze(1)).permute({1,2,0}) + G[0]*DvtDv
                          + bmm(fvv.unsqueeze(2),fuu.unsqueeze(1)).permute({1,2,0}) + G[3]*DutDu)
                  + 0.5*k[3]*(bmm(fuv.unsqueeze(2),fuv.unsqueeze(1)).permute({1,2,0}));
    // ignoring G[0][1]*(Du.t()*Dv+Dv.t()*Du)/2. term
    // because may not be positive definite
    return make_pair(-(batch_a*hess_e).permute({2,0,1}), -(batch_a*grad_e).t());
}

template <Space s>
Tensor bending_energy (const Edge *edge) {
    const Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
    if (!face0 || !face1)
        return ZERO;
    Tensor theta = dihedral_angle<s>(edge);
    Tensor a = face0->a + face1->a;
    const BendingData &bend0 = (*::materials)[face0->label]->bending,
                      &bend1 = (*::materials)[face1->label]->bending;
    Tensor ke = bending_stiffness(edge, bend0, bend1);
    Tensor weakening = max((*::materials)[face0->label]->weakening,
                           (*::materials)[face1->label]->weakening);
    ke = ke/(1 + weakening*edge->damage);
    Tensor shape = sq(edge->l)/(2*a);
    Tensor ans = ke*shape*sq(theta - edge->theta_ideal)/4;
    // cout << stack({edge->n[0]->x,edge->n[1]->x,edge_opp_vert(edge, 0)->node->x,edge_opp_vert(edge, 1)->node->x}) << endl;
    // cout << endl;
    return ans;
}

// Tensor distance (const Tensor &x, const Tensor &a, const Tensor &b) {
//     Tensor e = b-a;
//     Tensor xp = e*dot(e, x-a)/dot(e,e);
//     // return norm((x-a)-xp);
//     return max(norm((x-a)-xp), 1e-3*norm(e));
// }

// Tensor barycentric_weights (const Tensor &x, const Tensor &a, const Tensor &b) {
//     Tensor e = b-a;
//     Tensor t = dot(e, x-a)/dot(e,e);
//     return stack({1-t, t});
// }

template <Space s>
pair<Tensor,Tensor> bending_force (const Edge *edge) {
    const Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
    if (!face0 || !face1)
        return make_pair(torch::zeros({12,12},TNOPT), torch::zeros({12},TNOPT));
    Tensor theta = dihedral_angle<s>(edge);
    Tensor a = face0->a + face1->a;
    Tensor x0 = pos<s>(edge->n[0]),
         x1 = pos<s>(edge->n[1]),
         x2 = pos<s>(edge_opp_vert(edge, 0)->node),
         x3 = pos<s>(edge_opp_vert(edge, 1)->node);

    Tensor e = x1-x0, dote = dot(e,e), norme = 1e-3*sqrt(dote);
    Tensor t0 = dot(e, x2-x0)/dote, t1 = dot(e, x3-x0)/dote;
    Tensor h0 = max(norm(x2-x0-e*t0),norme), h1 = max(norm(x3-x0-e*t1),norme);
    Tensor n0 = nor<s>(face0)/h0, n1 = nor<s>(face1)/h1;
    Tensor w_f = stack({t0-1,t1-1,-t0,-t1,ONE,ZERO,ZERO,ONE}).reshape({4,2});
    Tensor dtheta = matmul(w_f, stack({n0,n1})).flatten();

    // Tensor h01 = distance(x2, x0, x1), h11 = distance(x3, x0, x1);
    // Tensor n01 = nor<s>(face0)/h01, n11 = nor<s>(face1)/h11;
    // Tensor w_f01 = barycentric_weights(x2, x0, x1),
    //      w_f11 = barycentric_weights(x3, x0, x1);
    // Tensor dtheta1 = cat({-(w_f01[0]*n01 + w_f11[0]*n11),
    //                                  -(w_f01[1]*n01 + w_f11[1]*n11),
    //                                  n01,
    //                                  n11});

    const BendingData &bend0 = (*::materials)[face0->label]->bending,
                      &bend1 = (*::materials)[face1->label]->bending;
    Tensor ke = bending_stiffness(edge, bend0, bend1);
    Tensor weakening = max((*::materials)[face0->label]->weakening,
                           (*::materials)[face1->label]->weakening);
    ke *= -sq(edge->l)/(4*a)/(1 + weakening*edge->damage);
    return make_pair(ke*ger(dtheta, dtheta),
                     ke*(theta - edge->theta_ideal)*dtheta);
}

pair<Tensor,Tensor> batch_bending_force (Tensor bat_x, Tensor bat_weak, Tensor bat_a, 
	Tensor bat_theta, Tensor bat_n, Tensor bat_bend, Tensor bat_ldaa, Tensor bat_bang, Tensor bat_theta_ideal,
    Tensor bat_oritheta) {
    Tensor e = bat_x[1]-bat_x[0], dote = sum(e*e, 1), norme = 1e-3*sqrt(dote);
    Tensor x2mx0 = bat_x[2]-bat_x[0], x3mx0 = bat_x[3]-bat_x[0];
    Tensor t0 = sum(e*(x2mx0), 1)/dote, t1 = sum(e*(x3mx0), 1)/dote;
    Tensor h0 = max(norm(x2mx0-e*t0.unsqueeze(1),2,{1}),norme), h1 = max(norm(x3mx0-e*t1.unsqueeze(1),2,{1}),norme);
    Tensor n0 = bat_n[0]/h0.unsqueeze(1), n1 = bat_n[1]/h1.unsqueeze(1);
    Tensor w_f = stack({t0-1,t1-1,-t0,-t1}).t().reshape({-1,2,2});
    Tensor dtheta = bmm(w_f, stack({n0,n1}, 1)).squeeze();//nx2x3
    dtheta = cat({dtheta,n0.unsqueeze(1),n1.unsqueeze(1)}, 1).reshape({-1,12});//nx12

    Tensor ke = batch_bending_stiffness(bat_oritheta*bat_ldaa*0.05, bat_bang, bat_bend);
    // int ind = 734;
    // cout << h0[ind].item<double>() << " "<<h1[ind].item<double>() << endl;
    // cout << n0[ind] << endl;
    // cout << n1[ind] << endl;
    // cout << w_f[ind] << endl;
    // cout << dtheta[ind] << endl;
    // cout << ke[ind] << endl;
    // cout << (bat_theta*bat_ldaa*0.05)[ind].item<double>() << endl;
    // cout << bat_ldaa[ind]<<endl;
    // cout << bat_theta[ind].item<double>() << endl;
    // cout << bat_bang[0][ind] << endl;
    // cout << bat_bang[1][ind] << endl;

    ke *= -sq(bat_ldaa)*bat_a/4/bat_weak;
    return make_pair(ke.unsqueeze(1).unsqueeze(1)*bmm(dtheta.unsqueeze(2), dtheta.unsqueeze(1)),
                     (ke*(bat_theta - bat_theta_ideal)).unsqueeze(1)*dtheta);
}

void add_submat (const Tensor &Asub, const vector<int> &ix, SpMat &A) {
    int m = ix.size();
    for (int i = 0; i < m; i++) {
        const Tensor &tmp = Asub.slice(0,i*3,i*3+3);
        for (int j = 0; j < m; j++)
            A(ix[i],ix[j]) += tmp.slice(1,j*3,j*3+3);
    }
}

void add_subvec (const Tensor &bsub, const vector<int> &ix, Tensor &b) {
    int m = ix.size();
    for (int i = 0; i < m; i++)
        b[ix[i]] += bsub.slice(0,i*3,i*3+3);//= b[ix[i]] + subvec3(bsub, i);
}

vector<int> indices (const Node *n0, const Node *n1, const Node *n2) {
    vector<int> ix(3);
    ix[0] = n0->index;
    ix[1] = n1->index;
    ix[2] = n2->index;
    return ix;
}

vector<int> indices (const Node *n0, const Node *n1,
                    const Node *n2, const Node *n3) {
    vector<int> ix(4);
    ix[0] = n0->index;
    ix[1] = n1->index;
    ix[2] = n2->index;
    ix[3] = n3->index;
    return ix;
}

template <Space s>
Tensor internal_energy (const Cloth &cloth) {
    // static int times = 0;++times;if (times==5)exit(0);
    const Mesh &mesh = cloth.mesh;
    ::materials = &cloth.materials;
    Tensor E = ZERO;
    for (int f = 0; f < mesh.faces.size(); f++)
        E = E + stretching_energy<s>(mesh.faces[f]);
    for (int e = 0; e < mesh.edges.size(); e++) {
        E = E + bending_energy<s>(mesh.edges[e]);
    }
//    cout << E.item<double>() << endl;
    return E;
}
template Tensor internal_energy<PS> (const Cloth&);
template Tensor internal_energy<WS> (const Cloth&);

// A = dt^2 J + dt damp J
// b = dt f + dt^2 J v + dt damp J v

template <Space s>
void add_internal_forces (const Cloth &cloth, SpMat &A,
                          Tensor &b, Tensor dt) {
//int ind=5;
    const Mesh &mesh = cloth.mesh;
    ::materials = &cloth.materials;
    vector<Tensor> batch_F, batch_stret, batch_weak, batch_a, batch_du, batch_dv;
    for (int f = 0; f < mesh.faces.size(); f++) {
        const Face* face = mesh.faces[f];
        const Node *n0 = face->v[0]->node, *n1 = face->v[1]->node,
                   *n2 = face->v[2]->node;
        Tensor F = derivative(pos<s>(face->v[0]->node), pos<s>(face->v[1]->node),
                pos<s>(face->v[2]->node), face);
        Tensor stret = (*::materials)[face->label]->stretching;
        Tensor weak = 1+(*::materials)[face->label]->weakening*face->damage;
        Tensor d = face->invDm.flatten();//acbd
        Tensor Du = cat({(-d[0]-d[2])*EYE3,d[0]*EYE3,d[2]*EYE3},1),//kronecker(rowmat(du), torch::eye(3,TNOPT)),
               Dv = cat({(-d[1]-d[3])*EYE3,d[1]*EYE3,d[3]*EYE3},1);//kronecker(rowmat(dv), torch::eye(3,TNOPT));
        batch_F.push_back(F);
        if (f==0) batch_stret.push_back(stret);
        batch_weak.push_back(weak);
        batch_du.push_back(Du);
        batch_dv.push_back(Dv);
        batch_a.push_back(face->a);
    }
    pair<Tensor,Tensor> membF = batch_stretching_force(stack(batch_F), stack(batch_stret), stack(batch_weak), stack(batch_du), stack(batch_dv), stack(batch_a));
//Tensor G = (bmm(stack(batch_F).permute({0,2,1}),stack(batch_F)) - 0.0*torch::eye(2,TNOPT).unsqueeze(0)).reshape({-1,4})*0.5;
//Tensor k = batch_stretching_stiffness(G.t(), stack(batch_stret));
//mesh.nodes[0]->x += (*::materials)[0]->stretching.sum()*torch::ones({3},TNOPT);
    for (int f = 0; f < mesh.faces.size(); ++f) {
        const Face* face = mesh.faces[f];
        const Node *n0 = face->v[0]->node, *n1 = face->v[1]->node,
                   *n2 = face->v[2]->node;
        Tensor vs = cat({n0->v, n1->v, n2->v});
        // pair<Tensor,Tensor> membF = stretching_force<s>(face);
        // Tensor J = membF.first;
        // Tensor F = membF.second;
        Tensor J = membF.first[f];
        Tensor F = membF.second[f];
        if ((dt == 0).item<int>()) {
            add_submat(-J, indices(n0,n1,n2), A);
            add_subvec(F, indices(n0,n1,n2), b);
        } else {
            Tensor damping = (*::materials)[face->label]->damping;
            add_submat(-dt*(dt+damping)*J, indices(n0,n1,n2), A);
            add_subvec(dt*(F + (dt+damping)*matmul(J,vs)), indices(n0,n1,n2), b);
        }
//        if (n0->index==ind||n1->index==ind||n2->index==ind){
//            double *p = F.data<double>();
//            for (int i = 0; i < 9; ++i) 
//                cout << p[i] << " ";
//            cout << endl;
//        }
    }
//return;
    map<int, int> edge_map;
    vector<Tensor> bat_x, bat_weak, bat_a, bat_theta, bat_n, bat_bend;
    vector<Tensor> bat_ldaa, bat_bang, bat_theta_ideal, bat_oritheta;
    for (int e = 0; e < mesh.edges.size(); e++) {
        const Edge *edge = mesh.edges[e];
        if (!edge->adjf[0] || !edge->adjf[1])
            continue;
        const Node *n0 = edge->n[0],
                   *n1 = edge->n[1],
                   *n2 = edge_opp_vert(edge, 0)->node,
                   *n3 = edge_opp_vert(edge, 1)->node;
        bat_x.push_back(pos<s>(n0));bat_x.push_back(pos<s>(n1));
        bat_x.push_back(pos<s>(n2));bat_x.push_back(pos<s>(n3));
        Tensor weak = 1+(*::materials)[edge->adjf[0]->label]->weakening*edge->damage;
        bat_weak.push_back(weak);
        bat_a.push_back(edge->adjf[0]->a+edge->adjf[1]->a);
        bat_theta.push_back(dihedral_angle<s>(edge));
        bat_n.push_back(nor<s>(edge->adjf[0]));
        bat_n.push_back(nor<s>(edge->adjf[1]));
        if(bat_a.size()==1)
            bat_bend.push_back((*::materials)[edge->adjf[0]->label]->bending);
        bat_ldaa.push_back(edge->ldaa);
        bat_bang.push_back(edge->bias_angle[0]);
        bat_bang.push_back(edge->bias_angle[1]);
        bat_theta_ideal.push_back(edge->theta_ideal);
        bat_oritheta.push_back(edge->theta);
        edge_map[e]=bat_a.size()-1;
    }
    pair<Tensor, Tensor> bendF = batch_bending_force(stack(bat_x).reshape({-1,4,3}).permute({1,0,2}), stack(bat_weak),
    	 stack(bat_a), stack(bat_theta), stack(bat_n).reshape({-1,2,3}).permute({1,0,2}), stack(bat_bend),
    	stack(bat_ldaa), stack(bat_bang).reshape({-1,2}).t(), stack(bat_theta_ideal), stack(bat_oritheta));
    for (int e = 0; e < mesh.edges.size(); ++e) {
    const Edge *edge = mesh.edges[e];
        if (!edge->adjf[0] || !edge->adjf[1])
            continue;
        const Node *n0 = edge->n[0],
                   *n1 = edge->n[1],
                   *n2 = edge_opp_vert(edge, 0)->node,
                   *n3 = edge_opp_vert(edge, 1)->node;
        //pair<Tensor,Tensor> bendF0 = bending_force<s>(edge);
        //Tensor J0 = bendF0.first;
        //Tensor F0 = bendF0.second;
        Tensor J = bendF.first[edge_map[e]];
        Tensor F = bendF.second[edge_map[e]];
        Tensor vs = cat({n0->v, n1->v, n2->v, n3->v});
        if ((dt == 0).item<int>()) {
            add_submat(-J, indices(n0,n1,n2,n3), A);
            add_subvec(F, indices(n0,n1,n2,n3), b);
        } else {
            Tensor damping = ((*::materials)[edge->adjf[0]->label]->damping +
                              (*::materials)[edge->adjf[1]->label]->damping)/2.;
            add_submat(-dt*(dt+damping)*J, indices(n0,n1,n2,n3), A);
            add_subvec(dt*(F + (dt+damping)*matmul(J,vs)), indices(n0,n1,n2,n3), b);
        }
//        if (n0->index==ind||n1->index==ind||n2->index==ind||n3->index==ind){
//            double *p = F.data<double>();
//            for (int i = 0; i < 12; ++i) 
//                cout << p[i] << " ";
//            cout << endl;
//            cout << n0->index<<" "<<n1->index<<" "<<n2->index<<" "<<n3->index<<" "<<endl;
        //     cout << n0->x[0].item<double>()<<" "<<n0->x[1].item<double>()<<" "<<n0->x[2].item<double>()<<" "<<endl;
        //     cout << n1->x[0].item<double>()<<" "<<n1->x[1].item<double>()<<" "<<n1->x[2].item<double>()<<" "<<endl;
        //     cout << n2->x[0].item<double>()<<" "<<n2->x[1].item<double>()<<" "<<n2->x[2].item<double>()<<" "<<endl;
        //     cout << n3->x[0].item<double>()<<" "<<n3->x[1].item<double>()<<" "<<n3->x[2].item<double>()<<" "<<endl;
        //     cout << edge_map[e] << endl;
//        }
    }
//cout << b[5] << endl;
}
template void add_internal_forces<PS> (const Cloth&, SpMat&,
                                       Tensor&, Tensor);
template void add_internal_forces<WS> (const Cloth&, SpMat&,
                                       Tensor&, Tensor);

bool contains (const Mesh &mesh, const Node *node) {
    return node->index < mesh.nodes.size() && mesh.nodes[node->index] == node;
}

Tensor constraint_energy (const vector<Constraint*> &cons) {
    Tensor E = ZERO;
    for (int c = 0; c < cons.size(); c++) {
        Tensor value = cons[c]->value();
        Tensor e = cons[c]->energy(value);
        E = E + e;
    }
    return E;
}

void add_constraint_forces (const Cloth &cloth, const vector<Constraint*> &cons,
                            SpMat &A, Tensor &b, Tensor dt) {
    const Mesh &mesh = cloth.mesh;
    for (int c = 0; c < cons.size(); c++) {
        Tensor value = cons[c]->value();
        Tensor g = cons[c]->energy_grad(value);
        Tensor h = cons[c]->energy_hess(value);
        MeshGrad grad = cons[c]->gradient();
        // f = -g*grad
        // J = -h*ger(grad,grad)
        Tensor v_dot_grad = ZERO;
        for (MeshGrad::iterator it = grad.begin(); it != grad.end(); it++) {
            const Node *node = it->first;
            v_dot_grad = v_dot_grad + dot(it->second, node->v);
        }
        for (MeshGrad::iterator it = grad.begin(); it != grad.end(); it++) {
            const Node *nodei = it->first;
            if (!contains(mesh, nodei))
                continue;
            int ni = nodei->index;
            for (MeshGrad::iterator jt = grad.begin(); jt != grad.end(); jt++) {
                const Node *nodej = jt->first;
                if (!contains(mesh, nodej))
                    continue;
                int nj = nodej->index;
                if ((dt == 0).item<int>())
                    add_submat(A, ni, nj, h*ger(it->second, jt->second));
                else
                    add_submat(A, ni, nj, dt*dt*h*ger(it->second, jt->second));
            }
            if ((dt == 0).item<int>())
                b[ni] = b[ni] - g*it->second;
            else
                b[ni] = b[ni] - dt*(g + dt*h*v_dot_grad)*it->second;
        }
    }
}

void add_friction_forces (const Cloth &cloth, const vector<Constraint*> cons,
                          SpMat &A, Tensor &b, Tensor dt) {
    const Mesh &mesh = cloth.mesh;
    for (int c = 0; c < cons.size(); c++) {
        MeshHess jac;
        MeshGrad force = cons[c]->friction(dt, jac);
        for (MeshGrad::iterator it = force.begin(); it != force.end(); it++) {
            const Node *node = it->first;
            if (!contains(mesh, node))
                continue;
            b[node->index] = b[node->index] + dt*it->second;
        }
        for (MeshHess::iterator it = jac.begin(); it != jac.end(); it++) {
            const Node *nodei = it->first.first, *nodej = it->first.second;
            if (!contains(mesh, nodei) || !contains(mesh, nodej))
                continue;
            add_submat(A, nodei->index, nodej->index, -dt*it->second);
        }
    }
}

void project_outside (Mesh &mesh, const vector<Constraint*> &cons);

void implicit_update (Cloth &cloth, const Tensor &fext,
                      const Tensor &Jext,
                      const vector<Constraint*> &cons, Tensor dt,
                      bool update_positions) {
    Mesh &mesh = cloth.mesh;
    vector<Vert*>::iterator vert_it;
    vector<Face*>::iterator face_it;
    int nn = mesh.nodes.size();
    // M Dv/Dt = F (x + Dx) = F (x + Dt (v + Dv))
    // Dv = Dt (M - Dt2 F)i F (x + Dt v)
    // A = M - Dt2 F
    // b = Dt F (x + Dt v)
    SpMat A(nn,nn);
    Tensor b = torch::zeros({nn,3}, TNOPT);
    for (int n = 0; n < mesh.nodes.size(); n++) {
        const Node* node = mesh.nodes[n];
        add_submat(A, n, n, torch::eye(3,TNOPT)*node->m - dt*dt*Jext[n]);
        b[n] = b[n] + dt*fext[n];
    }
    add_internal_forces<WS>(cloth, A, b, dt);
    add_constraint_forces(cloth, cons, A, b, dt);
    add_friction_forces(cloth, cons, A, b, dt);
    Tensor dv = taucs_linear_solve(A, b.reshape({nn*3})).reshape({nn,3});
    for (int n = 0; n < mesh.nodes.size(); n++) {
        Node *node = mesh.nodes[n];
        node->v = node->v + dv[n];
        if (n==0)
        if (update_positions)
            node->x = node->x + node->v*dt;
        node->acceleration = dv[n]/dt;
    }
    project_outside(cloth.mesh, cons);
    compute_ws_data(mesh);
}

Tensor wind_force (const Face *face, const Wind &wind) {
    Tensor vface = (face->v[0]->node->v + face->v[1]->node->v
                  + face->v[2]->node->v)/3.;
    Tensor vrel = wind.velocity - vface;
    Tensor vn = dot(face->n, vrel);
    Tensor vt = vrel - vn*face->n;
    return wind.density*face->a*abs(vn)*vn*face->n + wind.drag*face->a*vt;
}

void add_external_forces (const Cloth &cloth, const Tensor &gravity,
                          const Wind &wind, Tensor &fext,
                          Tensor &Jext) {
    const Mesh &mesh = cloth.mesh;
    for (int n = 0; n < mesh.nodes.size(); n++) {
        fext[n] = fext[n] + mesh.nodes[n]->m*gravity;
        // fext[n] = fext[n] + (mesh.nodes[4]->x-mesh.nodes[n]->x)*0.1;
    }
    for (int f = 0; f < mesh.faces.size(); f++) {
        const Face *face = mesh.faces[f];
        Tensor fw = wind_force(face, wind);
        for (int v = 0; v < 3; v++)
            fext[face->v[v]->node->index] = fext[face->v[v]->node->index] + fw/3.;
    }
}

void add_morph_forces (const Cloth &cloth, const Morph &morph, Tensor t,
                       Tensor dt, Tensor &fext, Tensor &Jext) {
    const Mesh &mesh = cloth.mesh;
    for (int v = 0; v < mesh.verts.size(); v++) {
        const Vert *vert = mesh.verts[v];
        Tensor x = morph.pos(t, vert->u);
        Tensor stiffness = exp(morph.log_stiffness.pos(t));
        Tensor n = vert->node->n;
        Tensor s = stiffness*vert->a;
        // // lower stiffness in tangential direction
        // Mat3x3 k = s*ger(n,n) + (s/10)*(Mat3x3(1) - ger(n,n));
        Tensor k = torch::eye(3,TNOPT)*s;
        Tensor c = sqrt(s*vert->m); // subcritical damping
        Tensor d = c/s * k;
        fext[vert->node->index] = fext[vert->node->index] - k*(vert->node->x - x);
        fext[vert->node->index] = fext[vert->node->index] - d*vert->node->v;
        Jext[vert->node->index] = Jext[vert->node->index] - (k + d/dt);
    }
}

void project_outside (Mesh &mesh, const vector<Constraint*> &cons) {
    int nn = mesh.nodes.size();
    Tensor w = torch::zeros({nn},TNOPT);
    Tensor dx = torch::zeros({nn,3},TNOPT);
    for (int c = 0; c < cons.size(); c++) {
        MeshGrad dxc = cons[c]->project();
        for (MeshGrad::iterator it = dxc.begin(); it != dxc.end(); it++) {
            const Node *node = it->first;
            Tensor wn = norm2(it->second);
            int n = node->index;
            if (n >= mesh.nodes.size() || mesh.nodes[n] != node)
                continue;
            w[n] = w[n] + wn;
            dx[n] = dx[n] + wn*it->second;
        }
    }
    for (int n = 0; n < nn; n++) {
        if ((w[n] == 0).item<int>())
            continue;
        mesh.nodes[n]->x = mesh.nodes[n]->x + dx[n]/w[n];
    }
}
