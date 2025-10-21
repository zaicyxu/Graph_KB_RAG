 
 /* Do Not Run the Following Code */
 /* Build Connection between service and certification */
/* Match (cert:Certification)
Create (service: Meta{Meta_Name: 'Service'})
Create (service)-[:Certify]->(cert) */



 /* Merge duplicate nodes */
/* MATCH (n:Meta)
WITH n.Meta_Name as name, COLLECT(n) AS ns
WHERE size(ns) > 1
CALL apoc.refactor.mergeNodes(ns) YIELD node
RETURN node */
 /* Do Not Run the Above Code */


/* Creating the relationships between manufacturer and other types  */LOAD CSV WITH HEADERS FROM 'file:///Entity_Relation_of_MSKG.csv' AS row

FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type = 'certification' and row.Entity_A_Type = 'manufacturer' THEN [1] ELSE [] END |
MERGE (manu:Manufactor {Id: row.Entity_A_Id ,Name:row.Entity_A, Node_Type: row.Entity_A_Type})
Merge (cert:Certification {Id: row.Entity_B_Id ,Name:row.Entity_B, Node_Type: row.Entity_B_Type})
merge (manu)-[:Certify]->(cert))
FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type = 'industry' and row.Entity_A_Type = 'manufacturer' THEN [1] ELSE [] END |
MERGE (manu:Manufactor {Id: row.Entity_A_Id ,Name:row.Entity_A, Node_Type: row.Entity_A_Type})
Merge (indu:Industry {Id: row.Entity_B_Id ,Name: row.Entity_B, Node_Type: row.Entity_B_Type})
merge (manu)-[:Belong]->(indu))
FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type = 'service' and row.Entity_A_Type = 'manufacturer' THEN [1] ELSE [] END |
MERGE (manu:Manufactor {Id: row.Entity_A_Id ,Name:row.Entity_A, Node_Type: row.Entity_A_Type})
Merge (busin:Business {Id: row.Entity_B_Id ,Name: row.Entity_B, Node_Type: row.Entity_B_Type})
merge (manu)-[:Work_on]->(busin))
FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type = 'material' and row.Entity_A_Type = 'manufacturer' THEN [1] ELSE [] END |
MERGE (manu:Manufactor {Id: row.Entity_A_Id ,Name:row.Entity_A, Node_Type: row.Entity_A_Type})
Merge (mater:Material {Id: row.Entity_B_Id ,Name: row.Entity_B, Node_Type: row.Entity_B_Type})
merge (manu)-[:Process]->(mater))


/* Creating the relationships between Industry and other types  */
LOAD CSV WITH HEADERS FROM 'file:///Entity_Relation_of_MSKG.csv' AS row 
FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type = 'industry' and row.Entity_A_Type = 'industry' THEN [1] ELSE [] END |
Merge (indu_1:Industry {Name: row.Entity_A, Id: row.Entity_A_Id, Node_Type: row.Entity_A_Type })
Merge (indu_2:Industry {Name: row.Entity_B, Id: row.Entity_B_Id, Node_Type: row.Entity_B_Type })
merge (indu_1)-[:Sub_Industry]->(indu_2))

/* Creating the relationships between Service and other types  */
LOAD CSV WITH HEADERS FROM 'file:///Entity_Relation_of_MSKG.csv' AS row 

FOREACH(ignoreMe IN CASE WHEN row.Entity_B_Type= 'service' and row.Entity_A_Type = 'service' THEN [1] ELSE [] END |
Merge (serv_1:Business {Name: row.Entity_A, Id: row.Entity_A_Id, Node_Type: row.Entity_A_Type })
Merge (serv_2:Business {Name: row.Entity_B, Id: row.Entity_B_Id, Node_Type: row.Entity_B_Type })
merge (serv_1)-[:Sub_Bussiness]->(serv_2))



/* Delete the redundant relationships between manufactor and the super business */
MATCH (a)-[shortcut:Work_on]->(:Business)<-[:Sub_Bussiness*1..]-(:Business)<-[:Work_on]-(a)
DELETE shortcut

/* Find the isolate industries */
match (m:Industry)
WHERE SIZE([(n:Industry)--(m:Industry) WHERE n <> m|1]) = 0
return m 




/* These are following test queries  */
MATCH p=(m:Manufactor{Name: 'berkseng.com'})-[]->() 
return p 


MATCH p=(m:Manufactor{Name: 'www.texmoprecisioncastings.com'})-[]->() 
RETURN p

/* Return relationships in depth wih multiple nodes*/
match (m:Manufactor)
with m limit 2
match p=((m)-[*1..]->())
return p 

/* 
Test Node Name
<elementId>: 1
<id>: 1
Id: 3850
Name: km-tool.com
Node_Type: manufacturer */

/* Generate Scenario-based Graphs */

call {

match p =(m:Manufactor)-[:Belong]->(m_1:Industry{Name: 'military'})
with *, size((m)-[:Belong]->()) as size_p
where size_p =1
match p_2 =(m:Manufactor)-[:Work_on]->(b_1)
with *, size((m)-[:Work_on]->()) as size_p
where size_p =1
return  m_1,m
}

call {

match p =(m_2:Manufactor)-[:Belong]->(m_1)
with *, size((m_2)-[:Belong]->()) as size_p
where size_p >1

return  m_2
}
match p = (m)-[r_1]-(), q =()<-[r_2]-(m_2)
where type(r_1) <> 'Belong' and  type(r_2) <> 'Belong'
return p,q 
ORDER BY RAND()
limit 20

