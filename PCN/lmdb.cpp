#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using std::string;
namespace db = caffe::db;

static int db_index = 0;

db::DB *LMDB__init_db(string path, string db_type)
{
	db::DB *db(db::GetDB(db_type));
	db->Open(path + db_type, db::NEW);

    return db;
}

void LMDB__fini_db(db::DB *db)
{
    db->Close();
}

void LMDB__add_to_database(db::DB *db, int datum_size, char *buffer, int label)
{
	Datum datum;
	db::Transaction *txn(db->NewTransaction());

	datum.set_channels(1);
	datum.set_height(1);
	datum.set_width(datum_size);

    datum.set_label(label);
    datum.set_data(buffer, datum_size);

    string out;
    datum.SerializeToString(&out);

    txn->Put(caffe::format_int(db_index, 5), out);
    db_index++;

	txn->Commit();
}
