#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/5 14:07
# @Author   : ReidChen
# Document  ：

from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687',
                              auth=('neo4j', '123456cctv@'))

# 添加关系函数
def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)

def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name=$name "
                         "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])
        
with driver.session() as session:
    session.write_transaction(add_friend, "Arthur", "Guinevere")
    session.write_transaction(add_friend, "Arthur", "Lancelot")
    session.write_transaction(add_friend, "Arthur", "Merlin")
    session.read_transaction(print_friends, "Arthur")
    
    
from py2neo import Graph, Node, Relationship
# 构造图
g = Graph('bolt://localhost:7687',username='neo4j',password='123456cctv@')
# 创建节点
tx = g.begin()
a = Node('Person', name='Alice')
tx.create(a)

b = Node('Person', name='Bob')
tx.create(b)
# 创建边
ab = Relationship(a, 'KNOWS', b)
# 运行
tx.create(ab)
tx.commit()

