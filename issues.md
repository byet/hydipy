- [ ] when you have evidence on continuous nodes with discrete parents (mixture nodes), you don't need discretization, you can just use likelihoods
- [ ] when you add evidence once, the states do not reset, they stay the same
- [ ] you can keep the discretizations in unevidenced model saved so you can start from there
- [ ] rather than building JT each time, you need to modify JT locally
- [ ] stopping rules need to be added
- [x] summary statistics for continuous nodes need to be added
- [x] continuous distribution parameters need to be added
- [x] statistical distributions that depend on other continuous nodes need to be added
- [ ] put continuous parent node into pgmpy shape
- [x] add parents to continuous nodes
- [x] add compute summary statistics method
- [x] build prob'da lowe bound upper bound bug var!
- [ ] partitioned expressionlarda distribution veya node girilebilmeli. Node girildiği durumda, o duruma koşullu kocaman bir CPT yapacak. Redundant node yaratmayacağız. Parent node'ların her biri bir durumda apply edecek şekilde kocaman distributionlar yaratacağız. 
- [ ] Normal expressionlar lazım? Bunları simulasyonla hazırlasak olmaz mı?
- [ ] Beta binomial lazım?
- [ ] I think we only merge the neighbor nodes with zero density