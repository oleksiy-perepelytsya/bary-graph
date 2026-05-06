from lib.config import Settings
from lib.db import get_collection

s = Settings.load()
coll = get_collection(s)

est = coll.estimated_document_count()
print(f"Estimated total: {est:,}")

print("\nBy doc_type + level:")
for r in coll.aggregate([
    {"$group": {"_id": {"doc_type": "$doc_type", "level": "$level"}, "count": {"$sum": 1}}},
    {"$sort": {"_id.doc_type": 1, "_id.level": 1}}
]):
    d = r["_id"]
    dt = d.get("doc_type", "?")
    lv = d.get("level", "?")
    print(f"  {dt} L{lv}:  {r['count']:,}")

print("\nNodes by node_type:")
for r in coll.aggregate([
    {"$match": {"doc_type": "node"}},
    {"$group": {"_id": "$node_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]):
    print(f"  {r['_id']}: {r['count']:,}")

print("\nL14 BaryEdges by edge_type:")
for r in coll.aggregate([
    {"$match": {"doc_type": "baryedge", "level": 14}},
    {"$group": {"_id": "$edge_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]):
    et = r["_id"] or "(null)"
    print(f"  {et}: {r['count']:,}")

print("\nL15 BaryEdges by edge_type (sample):")
for r in coll.aggregate([
    {"$match": {"doc_type": "baryedge", "level": 15}},
    {"$group": {"_id": "$edge_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]):
    et = r["_id"] or "(null)"
    print(f"  {et}: {r['count']:,}")

print("\nMetaBary BEs by level (L<=13):")
for r in coll.aggregate([
    {"$match": {"doc_type": "baryedge", "level": {"$lte": 13}}},
    {"$group": {"_id": "$level", "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}}
]):
    print(f"  L{r['_id']}: {r['count']:,}")

print("\nOrphans (parent_edge_id=None) by doc_type+level:")
for r in coll.aggregate([
    {"$match": {"parent_edge_id": None}},
    {"$group": {"_id": {"doc_type": "$doc_type", "level": "$level"}, "count": {"$sum": 1}}},
    {"$sort": {"_id.doc_type": 1, "_id.level": 1}}
]):
    d = r["_id"]
    dt = d.get("doc_type", "?")
    lv = d.get("level", "?")
    print(f"  {dt} L{lv}:  {r['count']:,}")

print("\nL14+L15 BEs by source:")
for r in coll.aggregate([
    {"$match": {"doc_type": "baryedge", "level": {"$in": [14, 15]}}},
    {"$group": {"_id": "$source", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]):
    src = r["_id"] or "(null)"
    print(f"  {src}: {r['count']:,}")
